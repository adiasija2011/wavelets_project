import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import INFO, __dict__ as medmnist_datasets

def get_medmnist_dataloader(dataset_flag, split, batch_size, download=True, num_workers=4):
    if dataset_flag not in INFO:
        raise ValueError(f"Invalid dataset flag '{dataset_flag}'. Valid flags: {list(INFO.keys())}")

    info = INFO[dataset_flag]
    dataset_class_name = info['python_class']
    if dataset_class_name not in medmnist_datasets:
        raise ValueError(f"Dataset class '{dataset_class_name}' not found in MedMNIST module.")

    dataset_class = medmnist_datasets[dataset_class_name]
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = dataset_class(split=split, transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)
    return loader, len(info['label'])
