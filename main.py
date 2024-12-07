import torch
import wandb
from data_loader import get_medmnist_dataloader
from model import WaveletCNNModel
from training import train_model
from evaluation import evaluate_model

# Initialize WandB
wandb.init(project="wavelet-cnn")

# Dataset and device setup
dataset_flag = "pathmnist"
batch_size = 32
num_epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, num_classes = get_medmnist_dataloader(dataset_flag, "train", batch_size)
val_loader, _ = get_medmnist_dataloader(dataset_flag, "val", batch_size)
dataloaders = {"train": train_loader, "val": val_loader}

print(f"len train loader:{len(train_loader)} , validation loader :{len(val_loader)}")
# Initialize model
model = WaveletCNNModel(num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train model
model = train_model(model, dataloaders, criterion, optimizer, num_epochs, device)

# Evaluate model
evaluate_model(model, val_loader, device)

# Finish WandB
wandb.finish()
