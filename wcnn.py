#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall
from torch.utils.data import DataLoader
import time
import copy
import os
import wandb
from medmnist import INFO
from medmnist import PathMNIST
from torchvision import transforms


# Initialize Weights and Biases
wandb.init(project="wavelet-cnn")

# Define the Wavelet Transform functions
class WaveletTransform(nn.Module):
    def __init__(self):
        super(WaveletTransform, self).__init__()
    
    def forward(self, batch_image):
        # batch_image shape: (batch_size, channels, height, width)
        # Assuming channels = 3

        # Split into R, G, B channels
        r = batch_image[:, 0, :, :]
        g = batch_image[:, 1, :, :]
        b = batch_image[:, 2, :, :]

        # Level 1 decomposition
        r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH = self.wavelet_decompose(r)
        g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH = self.wavelet_decompose(g)
        b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH = self.wavelet_decompose(b)

        wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH, 
                        g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
                        b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
        transform_batch = torch.stack(wavelet_data, dim=1)  # shape: (batch_size, 12, h, w)

        # Level 2 decomposition
        r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2 = self.wavelet_decompose(r_wavelet_LL)
        g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2 = self.wavelet_decompose(g_wavelet_LL)
        b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2 = self.wavelet_decompose(b_wavelet_LL)

        wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2, 
                           g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
                           b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
        transform_batch_l2 = torch.stack(wavelet_data_l2, dim=1)

        # Level 3 decomposition
        r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3 = self.wavelet_decompose(r_wavelet_LL2)
        g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3 = self.wavelet_decompose(g_wavelet_LL2)
        b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3 = self.wavelet_decompose(b_wavelet_LL2)

        wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3, 
                           g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3,
                           b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3]
        transform_batch_l3 = torch.stack(wavelet_data_l3, dim=1)

        # Level 4 decomposition
        r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4 = self.wavelet_decompose(r_wavelet_LL3)
        g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4 = self.wavelet_decompose(g_wavelet_LL3)
        b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4 = self.wavelet_decompose(b_wavelet_LL3)

        wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4, 
                           g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4,
                           b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4]
        transform_batch_l4 = torch.stack(wavelet_data_l4, dim=1)

        return [transform_batch, transform_batch_l2, transform_batch_l3, transform_batch_l4]
    
    def wavelet_decompose(self, channel):
        # channel shape: (batch_size, h, w)
        wavelet_L, wavelet_H = self.WaveletTransformAxisY(channel)
        wavelet_LL, wavelet_LH = self.WaveletTransformAxisX(wavelet_L)
        wavelet_HL, wavelet_HH = self.WaveletTransformAxisX(wavelet_H)
        return wavelet_LL, wavelet_LH, wavelet_HL, wavelet_HH

    def WaveletTransformAxisY(self, batch_img):
        # batch_img shape: (batch_size, h, w)
        odd_img  = batch_img[:, 0::2, :]
        even_img = batch_img[:, 1::2, :]
        L = (odd_img + even_img) / 2.0
        H = torch.abs(odd_img - even_img)
        return L, H

    def WaveletTransformAxisX(self, batch_img):
        # batch_img shape: (batch_size, h, w)
        # transpose + flip left-right
        tmp_batch = torch.flip(batch_img.transpose(1, 2), [2])
        dst_L, dst_H = self.WaveletTransformAxisY(tmp_batch)
        # transpose + flip up-down
        dst_L = torch.flip(dst_L.transpose(1, 2), [1])
        dst_H = torch.flip(dst_H.transpose(1, 2), [1])
        return dst_L, dst_H

# Define the model architecture
class WaveletCNNModel(nn.Module):
    def __init__(self, num_classes=12):
        super(WaveletCNNModel, self).__init__()
        self.wavelet = WaveletTransform()
        # Level 1
        self.conv_1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.norm_1 = nn.BatchNorm2d(64)
        
        self.conv_1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.norm_1_2 = nn.BatchNorm2d(64)
        
        # Level 2
        self.conv_a = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.norm_a = nn.BatchNorm2d(64)
        
        # After concat level 2
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm_2 = nn.BatchNorm2d(128)
        
        self.conv_2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.norm_2_2 = nn.BatchNorm2d(128)
        
        # Level 3
        self.conv_b = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.norm_b = nn.BatchNorm2d(64)
        
        self.conv_b_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm_b_2 = nn.BatchNorm2d(128)
        
        # After concat level 3
        self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm_3 = nn.BatchNorm2d(256)
        
        self.conv_3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.norm_3_2 = nn.BatchNorm2d(256)
        
        # Level 4
        self.conv_c = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.norm_c = nn.BatchNorm2d(64)
        
        self.conv_c_2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.norm_c_2 = nn.BatchNorm2d(256)
        
        self.conv_c_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm_c_3 = nn.BatchNorm2d(256)
        
        # After concat level 4
        self.conv_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.norm_4 = nn.BatchNorm2d(256)
        
        self.conv_4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.norm_4_2 = nn.BatchNorm2d(256)
        
        # Final layers
        self.conv_5_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.norm_5_1 = nn.BatchNorm2d(128)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        
        self.fc_5 = nn.Linear(128 * 7 * 7, 1024)
        self.norm_5 = nn.BatchNorm1d(1024)
        self.drop_5 = nn.Dropout(0.5)
        
        self.output = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224)
        input_l1, input_l2, input_l3, input_l4 = self.wavelet(x)
        # Level 1
        x1 = F.relu(self.norm_1(self.conv_1(input_l1)))
        x1 = F.relu(self.norm_1_2(self.conv_1_2(x1)))
        
        # Level 2
        x2 = F.relu(self.norm_a(self.conv_a(input_l2)))
        
        # Concatenate level 1 and 2
        x12 = torch.cat([x1, x2], dim=1)  # dim=1 is the channel dimension
        x12 = F.relu(self.norm_2(self.conv_2(x12)))
        x12 = F.relu(self.norm_2_2(self.conv_2_2(x12)))
        
        # Level 3
        x3 = F.relu(self.norm_b(self.conv_b(input_l3)))
        x3 = F.relu(self.norm_b_2(self.conv_b_2(x3)))
        
        # Concatenate level 2 and 3
        x123 = torch.cat([x12, x3], dim=1)
        x123 = F.relu(self.norm_3(self.conv_3(x123)))
        x123 = F.relu(self.norm_3_2(self.conv_3_2(x123)))
        
        # Level 4
        x4 = F.relu(self.norm_c(self.conv_c(input_l4)))
        x4 = F.relu(self.norm_c_2(self.conv_c_2(x4)))
        x4 = F.relu(self.norm_c_3(self.conv_c_3(x4)))
        
        # Concatenate level 3 and 4
        x1234 = torch.cat([x123, x4], dim=1)
        x1234 = F.relu(self.norm_4(self.conv_4(x1234)))
        x1234 = F.relu(self.norm_4_2(self.conv_4_2(x1234)))
        
        x5 = F.relu(self.norm_5_1(self.conv_5_1(x1234)))
        x5 = self.avg_pool(x5)  # Output shape: (batch_size, 128, 7, 7)
        x5 = x5.view(x5.size(0), -1)  # Flatten
        x5 = F.relu(self.norm_5(self.fc_5(x5)))
        x5 = self.drop_5(x5)
        output = self.output(x5)
        return output

def get_medmnist_dataloader(dataset_flag, split, batch_size, download=True, num_workers=4):
    from medmnist import INFO, __dict__ as medmnist_datasets  # Dynamically load dataset classes
    
    if dataset_flag not in INFO:
        raise ValueError(f"Dataset flag '{dataset_flag}' is invalid. Valid flags: {list(INFO.keys())}")
    
    info = INFO[dataset_flag]
    dataset_class_name = info['python_class']

    # Ensure the dataset class exists in medmnist
    if dataset_class_name not in medmnist_datasets:
        raise ValueError(f"Dataset class '{dataset_class_name}' not found in MedMNIST module.")
    
    dataset_class = medmnist_datasets[dataset_class_name]
    
    # Define transform
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = dataset_class(split=split, transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)
    return loader, len(info['label'])


def train_model(model, dataloaders, criterion, optimizer, num_epochs=20, device="cuda"):
    since = time.time()
    val_acc_history = []
    train_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize metrics
    accuracy = Accuracy(task='multiclass', num_classes=model.num_classes).to(device)
    precision = Precision(average='macro', num_classes=model.num_classes).to(device)
    recall = Recall(average='macro', num_classes=model.num_classes).to(device)

    # Track metrics with Weights and Biases
    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        epoch_metrics = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Log metrics
            epoch_metrics[f"{phase}_loss"] = epoch_loss
            epoch_metrics[f"{phase}_accuracy"] = epoch_acc.item()

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "best_model.pth")

            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        # Log metrics to Weights and Biases
        wandb.log(epoch_metrics)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history


# Evaluate model
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

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Save predictions and labels for offline analysis
    os.makedirs(save_path, exist_ok=True)
    torch.save(all_preds, os.path.join(save_path, "predictions.pt"))
    torch.save(all_labels, os.path.join(save_path, "labels.pt"))

# Main setup for MedMNIST training
dataset_flag = "pathmnist"  # Change as needed
batch_size = 32
num_epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prepare data loaders
train_loader, num_classes = get_medmnist_dataloader(dataset_flag, "train", batch_size)
val_loader, _ = get_medmnist_dataloader(dataset_flag, "val", batch_size)
dataloaders = {"train": train_loader, "val": val_loader}

# Initialize model, criterion, optimizer
model = WaveletCNNModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train model
model, val_acc_history, train_acc_history = train_model(
    model, dataloaders, criterion, optimizer, num_epochs, device
)

# Save training history for offline analysis
torch.save({"train_acc": train_acc_history, "val_acc": val_acc_history}, "training_history.pt")

# Evaluate model
evaluate_model(model, val_loader, device)

# Finish Weights and Biases run
wandb.finish()

