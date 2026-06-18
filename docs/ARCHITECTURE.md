# Architecture

## System Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Mobile/Edge    │────▶│  FastAPI Backend │────▶│  React Frontend │
│  Camera Input   │     │  (detection/seg) │     │  Visualization  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                        │
         ▼                        ▼
  Faster R-CNN               CraSAM / U-Net
  Transfer Learning          Prompt Adapter
```

## Pipeline

1. **Image Acquisition**: Industrial cameras capture high-resolution images.
2. **Preprocessing**: Resize, normalize, and augment.
3. **Detection**: Faster R-CNN locates crack regions and classifies width grade.
4. **Segmentation**: CraSAM generates pixel-level crack masks.
5. **Parameter Extraction**: Skeletonization + distance transform for length/width/angle.
6. **Health Assessment**: Fractal dimension + TCI mapping.
7. **Reporting**: JSON report + crack expansion map.

## Model Zoo

See `README.md` for the model zoo table.
