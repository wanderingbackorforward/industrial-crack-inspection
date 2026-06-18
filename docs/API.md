# API Documentation

## Endpoints

### `GET /health`
Returns service status and device info.

### `POST /predict`
Upload an image and receive detection + segmentation results.

**Parameters**:
- `file`: image file
- `pixel_size_mm`: physical size of one pixel in millimeters

**Response**:
```json
{
  "detections": [
    {"box": [x1, y1, x2, y2], "score": 0.92, "label": 1}
  ],
  "parameters": {
    "length_mm": 120.5,
    "mean_width_mm": 0.25,
    "max_width_mm": 0.60,
    "angle_deg": 35.0
  },
  "fractal_dimension": 1.1849
}
```
