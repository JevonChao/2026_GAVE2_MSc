import torch


class SegmentationMetric(object):
    """
    Segmentation evaluation metrics.

    Calculate:
        Dice
        IoU
        Precision
        Recall
        Specificity

    Input:
        pred : network output (logits)
        target : ground truth mask (0/1)

    Example:
        metric = SegmentationMetric()

        dice, iou, precision, recall, specificity = metric(pred, target)
    """

    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, pred, target):

        # logits -> probability
        pred = torch.sigmoid(pred)

        # probability -> binary mask
        pred = (pred > self.threshold).float()

        target = target.float()

        # flatten
        pred = pred.view(-1)
        target = target.view(-1)

        TP = (pred * target).sum()

        FP = (pred * (1 - target)).sum()

        FN = ((1 - pred) * target).sum()

        TN = ((1 - pred) * (1 - target)).sum()

        dice = (2 * TP + self.smooth) / (
            2 * TP + FP + FN + self.smooth
        )

        iou = (TP + self.smooth) / (
            TP + FP + FN + self.smooth
        )

        precision = (TP + self.smooth) / (
            TP + FP + self.smooth
        )

        recall = (TP + self.smooth) / (
            TP + FN + self.smooth
        )

        specificity = (TN + self.smooth) / (
            TN + FP + self.smooth
        )

        return (
            dice.item(),
            iou.item(),
            precision.item(),
            recall.item(),
            specificity.item()
        )