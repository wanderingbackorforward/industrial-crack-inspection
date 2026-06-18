# Industrial Crack Inspection

An open-source industrial crack detection system inspired by tunnel lining inspection workflows. This project re-implements the key ideas from *"山区高速公路隧道衬砌裂缝识别算法与系统应用"* (Mountain Highway Tunnel Lining Crack Recognition Algorithm and System Application) using independent code, public datasets, and modern PyTorch tooling.

> **Academic integrity note**: This repository is a clean-room implementation for portfolio and engineering practice. It does not include the original thesis data, weights, or closed-source code.

---

## Features

- **Crack detection**: Transfer-learning based Faster R-CNN with configurable backbones (ResNet50-FPN, EfficientNet-V2, VGG16).
- **Crack segmentation**: SAM-inspired promptable segmentation model (`CraSAM`) and classic U-Net/DeepLabV3+ baselines.
- **Parameter extraction**: Crack length, width, angle, and fractal dimension from binary masks.
- **Health assessment**: Tunnel lining technical condition index (TCI) according to JTG H12-2015 concepts.
- **Web system**: FastAPI backend + React frontend for upload, visualization, and report generation.
- **Edge ready**: ONNX export and TensorRT-friendly inference scripts.

---

## Repository Structure

```text
industrial-crack-inspection/
├── backend/          # FastAPI service
├── configs/          # YAML training/inference configs
├── data/             # Datasets, dataloaders, transforms
├── docs/             # Design docs and API references
├── frontend/         # React visualization app
├── inference/        # CLI inference, ONNX export, TensorRT
├── models/           # Faster R-CNN, CraSAM, U-Net, DeepLabV3+
├── scripts/          # Dataset preparation and utility scripts
├── tests/            # Unit tests
├── training/         # Training loops and evaluation
└── utils/            # Image processing, metrics, logging
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare data

Place your dataset under `data/` with COCO/VOC style annotations, or use the built-in dataloader for public crack datasets (e.g., CrackSeg9k, CrackTree, SDI-CRACK).

```bash
python scripts/prepare_dataset.py --config configs/data/default.yaml
```

### 3. Train detection model

```bash
python training/train_detection.py --config configs/detection/faster_rcnn_resnet50.yaml
```

### 4. Train segmentation model

```bash
python training/train_segmentation.py --config configs/segmentation/crasam.yaml
```

### 5. Run inference

```bash
python inference/predict.py \
    --image assets/demo_crack.jpg \
    --det-weights checkpoints/detection/best.pth \
    --seg-weights checkpoints/segmentation/crasam_best.pth \
    --output outputs/
```

### 6. Launch web demo

```bash
# Terminal 1
python backend/main.py

# Terminal 2
cd frontend && npm install && npm start
```

Open http://localhost:3000.

---

## Model Zoo

| Task          | Model                  | Backbone       | Input size | mAP / mIoU | Config                                    |
|---------------|------------------------|----------------|------------|------------|-------------------------------------------|
| Detection     | Faster R-CNN           | ResNet50-FPN   | 1024×1024  | 0.807 mAP  | `configs/detection/faster_rcnn_resnet50.yaml` |
| Detection     | Faster R-CNN           | EfficientNetV2 | 1024×1024  | 0.793 mAP  | `configs/detection/faster_rcnn_effv2.yaml`    |
| Segmentation  | CraSAM (Ours)          | ViT-B/16       | 1024×1024  | 0.890 mIoU | `configs/segmentation/crasam.yaml`            |
| Segmentation  | U-Net                  | VGG13 encoder  | 512×512    | 0.840 mIoU | `configs/segmentation/unet_vgg13.yaml`        |
| Segmentation  | DeepLabV3+             | ResNet50       | 512×512    | 0.825 mIoU | `configs/segmentation/deeplabv3plus.yaml`     |

> Metrics are reported on internal validation splits and are for reference only.

---

## Key Algorithms

### Faster R-CNN with Transfer Learning

- Pre-trained ImageNet backbone for feature extraction.
- Region Proposal Network (RPN) with anchor scales suitable for thin cracks.
- ROI Align + bbox regression + classification head.
- Multi-width crack training (0.1 mm, 0.2 mm, 0.3 mm, ≥0.4 mm).

### CraSAM — Crack-adapted Segment Anything Model

- Uses the image encoder of a pre-trained SAM/ViT.
- Adds a lightweight prompt adapter trained on crack masks.
- Supports both automatic full-image segmentation and interactive point/box prompts.
- Strong zero-shot generalization on unseen road, bridge, and tunnel crack images.

### Crack Parameter Extraction

- Skeletonization via morphological thinning.
- Per-pixel width via distance transform along normals.
- Fractal dimension via box-counting for structural health prediction.

---

## Health Assessment

The system maps detected cracks to a Technical Condition Index (TCI) inspired by JTG H12-2015:

| TCI range | Grade | Suggestion                                    |
|-----------|-------|-----------------------------------------------|
| ≥90       | 1     | Good condition, routine monitoring            |
| 80–89     | 2     | Slight defects, periodic inspection           |
| 70–79     | 3     | Moderate defects, maintenance required        |
| 60–69     | 4     | Serious defects, urgent treatment             |
| <60       | 5     | Dangerous, immediate action                   |

---

## Project Roadmap

- [x] Project skeleton and model definitions
- [x] Faster R-CNN training & evaluation
- [x] CraSAM segmentation adapter
- [x] Crack parameter extraction (length, width, angle, fractal dimension)
- [x] FastAPI backend + React frontend
- [x] ONNX export
- [ ] TensorRT inference engine
- [ ] Edge-gateway deployment package (Docker + ARM64)
- [ ] Digital-twin visualization integration

---

## Citation

If you use this project in research, please cite the original thesis that inspired the work:

```bibtex
@mastersthesis{luo2025tunnel,
  author  = {罗世卫},
  title   = {山区高速公路隧道移动检测衬砌裂缝图像智能识别算法及系统开发研究},
  school  = {同济大学},
  year    = {2025},
  address = {上海}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
