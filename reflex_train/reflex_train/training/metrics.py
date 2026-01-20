import torch


class RunningMetrics:
    def __init__(self):
        self.loss_sum = 0.0
        self.batches = 0
        self.key_tp = 0
        self.key_fp = 0
        self.key_fn = 0

    def update_loss(self, loss_val):
        self.loss_sum += loss_val
        self.batches += 1

    def update_keys(self, logits, targets, threshold):
        preds = torch.sigmoid(logits) > threshold
        targs = targets > 0.5
        self.key_tp += (preds & targs).sum().item()
        self.key_fp += (preds & ~targs).sum().item()
        self.key_fn += (~preds & targs).sum().item()

    def summary(self):
        loss = self.loss_sum / self.batches if self.batches else 0.0
        precision = (
            self.key_tp / (self.key_tp + self.key_fp)
            if (self.key_tp + self.key_fp) > 0
            else 0.0
        )
        recall = (
            self.key_tp / (self.key_tp + self.key_fn)
            if (self.key_tp + self.key_fn) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "loss": loss,
            "key_precision": precision,
            "key_recall": recall,
            "key_f1": f1,
        }
