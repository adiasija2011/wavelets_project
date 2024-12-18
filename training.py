import torch
import time
import copy
from torchmetrics import Accuracy, Precision, Recall
import wandb

def train_model(model, dataloaders, criterion, optimizer, num_epochs=20, device="cuda"):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Initialize epoch loss and accuracy trackers for WandB
        epoch_loss = {'train': 0.0, 'val': 0.0}
        epoch_accuracy = {'train': 0.0, 'val': 0.0}
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # Flatten the labels if they have an extra dimension
                if labels.ndim > 1:
                    labels = labels.squeeze(dim=1)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch loss and accuracy
            epoch_loss[phase] = running_loss / len(dataloaders[phase].dataset)
            epoch_accuracy[phase] = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f"{phase} Loss: {epoch_loss[phase]:.4f}, Accuracy: {epoch_accuracy[phase]:.4f}")

            # Save the best model weights for validation
            if phase == 'val' and epoch_accuracy[phase] > best_acc:
                best_acc = epoch_accuracy[phase]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "best_model.pth")

        # Log metrics to WandB after each epoch
        wandb.log({
            "train_loss": epoch_loss['train'],
            "val_loss": epoch_loss['val'],
            "train_accuracy": epoch_accuracy['train'],
            "val_accuracy": epoch_accuracy['val'],
            "epoch": epoch + 1
        })

    model.load_state_dict(best_model_wts)
    return model
