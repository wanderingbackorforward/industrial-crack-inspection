"""FastAPI backend for industrial crack inspection."""
import os
import io
import json
import tempfile
from typing import List
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_faster_rcnn, build_crasam
from utils import load_config, extract_crack_params, box_counting_dimension


app = FastAPI(title="Industrial Crack Inspection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DET_CFG = load_config("configs/detection/faster_rcnn_resnet50.yaml")
SEG_CFG = load_config("configs/segmentation/crasam.yaml")

DET_MODEL = build_faster_rcnn(DET_CFG).to(DEVICE)
SEG_MODEL = build_crasam(SEG_CFG).to(DEVICE)

# In a real deployment load pretrained weights here:
# DET_MODEL.load_state_dict(torch.load("checkpoints/detection/best.pth", map_location=DEVICE))
# SEG_MODEL.load_state_dict(torch.load("checkpoints/segmentation/crasam_best.pth", map_location=DEVICE))


def _preprocess(image: Image.Image, size: int):
    img = image.convert("RGB").resize((size, size))
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), pixel_size_mm: float = 1.0):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    det_tensor = _preprocess(image, DET_CFG["data"]["image_size"])
    seg_tensor = _preprocess(image, SEG_CFG["data"]["image_size"])

    DET_MODEL.eval()
    SEG_MODEL.eval()
    with torch.no_grad():
        det_out = DET_MODEL([det_tensor.squeeze(0).to(DEVICE)])[0]
        seg_logits = SEG_MODEL(seg_tensor.to(DEVICE))
        seg_mask = (seg_logits.sigmoid().squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    keep = det_out["scores"] > 0.5
    boxes = det_out["boxes"][keep].cpu().numpy().tolist()
    scores = det_out["scores"][keep].cpu().numpy().tolist()
    labels = det_out["labels"][keep].cpu().numpy().tolist()

    params = extract_crack_params(seg_mask, pixel_size_mm=pixel_size_mm)
    fractal_dim = box_counting_dimension(seg_mask)

    # Build overlay image
    overlay = np.array(image.resize((1024, 1024)))
    overlay[seg_mask > 0] = [255, 0, 0]
    overlay_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    buf.seek(0)

    return JSONResponse({
        "detections": [{"box": b, "score": s, "label": l} for b, s, l in zip(boxes, scores, labels)],
        "parameters": params,
        "fractal_dimension": float(fractal_dim),
        "overlay_url": "/predict/overlay",
    })


@app.get("/predict/overlay")
def get_overlay():
    # Simplified placeholder endpoint; real implementation stores latest overlay.
    return {"detail": "Use /predict which returns overlay_url conceptually."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
