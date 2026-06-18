"""Extract geometric parameters and fractal dimension from crack masks."""
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


def extract_crack_params(mask: np.ndarray, pixel_size_mm: float = 1.0):
    """Return length (mm), mean width (mm), max width (mm), and dominant angle (deg)."""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return {"length_mm": 0.0, "mean_width_mm": 0.0, "max_width_mm": 0.0, "angle_deg": 0.0}

    # Skeleton length
    skel = skeletonize(mask > 0)
    length_px = np.sum(skel)

    # Width via distance transform
    dist = distance_transform_edt(mask)
    widths_px = 2 * dist[skel > 0]
    mean_width_px = float(np.mean(widths_px)) if widths_px.size else 0.0
    max_width_px = float(np.max(widths_px)) if widths_px.size else 0.0

    # Dominant angle via PCA of crack pixels
    ys, xs = np.where(mask > 0)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(points) >= 2:
        _, eigenvectors = cv2.PCACompute(points, mean=None)
        vec = eigenvectors[0]
        angle = np.degrees(np.arctan2(abs(vec[1]), abs(vec[0])))
    else:
        angle = 0.0

    return {
        "length_mm": float(length_px * pixel_size_mm),
        "mean_width_mm": float(mean_width_px * pixel_size_mm),
        "max_width_mm": float(max_width_px * pixel_size_mm),
        "angle_deg": float(angle),
    }


def box_counting_dimension(mask: np.ndarray, min_box=2, max_box=None):
    """Estimate fractal dimension using box counting."""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return 0.0
    H, W = mask.shape
    max_box = max_box or min(H, W) // 2
    sizes = []
    counts = []
    size = min_box
    while size <= max_box:
        counts.append(_box_count(mask, size))
        sizes.append(size)
        size *= 2
    if len(sizes) < 2:
        return 0.0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def _box_count(mask, box_size):
    h, w = mask.shape
    count = 0
    for y in range(0, h, box_size):
        for x in range(0, w, box_size):
            if mask[y:y+box_size, x:x+box_size].sum() > 0:
                count += 1
    return max(count, 1)
