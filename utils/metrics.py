"""Evaluation metrics for detection and segmentation."""
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import BinaryF1Score


def detection_metrics(predictions, targets, iou_thresholds=None):
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
    )
    metric.update(predictions, targets)
    return metric.compute()


def segmentation_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """preds and targets: (B, H, W) logits or probabilities."""
    binary_preds = (preds.sigmoid() > threshold).long()
    targets = targets.long()
    miou_metric = MeanIoU(num_classes=2)
    f1_metric = BinaryF1Score()
    miou = miou_metric(binary_preds, targets)
    f1 = f1_metric(binary_preds.view(-1), targets.view(-1))
    return {"miou": miou.item(), "f1": f1.item()}
