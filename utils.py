# utils.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(base_dir, batch_size=16):
    folds = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    loaders = []
    class_names = None

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    for fold in folds:
        dataset = datasets.ImageFolder(os.path.join(base_dir, fold), transform=transform)
        if class_names is None:
            class_names = dataset.classes
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)

    return loaders, len(class_names), class_names
