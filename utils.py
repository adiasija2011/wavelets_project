from torchmetrics import Accuracy, Precision, Recall

def initialize_metrics(num_classes, device):
    return {
        "accuracy": Accuracy(task='multiclass', num_classes=num_classes).to(device),
        "precision": Precision(average='macro', num_classes=num_classes).to(device),
        "recall": Recall(average='macro', num_classes=num_classes).to(device),
    }
