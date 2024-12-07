import os
import torch

def evaluate_model(model, dataloader, device, save_path="results"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    os.makedirs(save_path, exist_ok=True)
    torch.save(torch.cat(all_preds), os.path.join(save_path, "predictions.pt"))
    torch.save(torch.cat(all_labels), os.path.join(save_path, "labels.pt"))
