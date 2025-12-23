import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set a global seed for reproducibility
SEED = 42

# Ensuring repeatablility across runs in PyTorch for consistent results.


def set_seed(seed=SEED):
    """Seed Python, NumPy, and PyTorch (CPU & all GPUs) and
    make cuDNN run in deterministic mode to ensure reproducibility."""

    # ---- Seed Python's built-in random module -----------------------
    random.seed(seed)
    # ---- Seed NumPy's random number generator -----------------------
    np.random.seed(seed)

    # ---- Seed PyTorch (CPU & all GPUs) ------------------------------
    torch.manual_seed(seed)            # Seed for CPU operations
    torch.cuda.manual_seed_all(seed)   # Seed for all GPU operations

    # ---- cuDNN: Configure for repeatable convolutions ---------------
    # This ensures that cudnn algorithms are deterministic.
    torch.backends.cudnn.deterministic = True
    # Disable cuDNN benchmarking to ensure consistent execution speed (can be slower).
    torch.backends.cudnn.benchmark = False

# Define worker_init_fn function for DataLoader workers.


def worker_init_fn(worker_id):
    """Re-seed each DataLoader worker so their RNGs don't collide.
    This ensures that each worker gets a unique, but reproducible, sequence of random numbers."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# Custom PyTorch Dataset class for the NEU-DET dataset that automatically maps subfolder names to class labels and loads images with optional transformations.
class NEUDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Get the class names
        self.classes = sorted(os.listdir(img_dir))
        # Create a mapping: {'crazing': 0, 'inclusion': 1, ......}
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        # Create a list of every single image path and its label
        self.images = []
        for class_name in self.classes:
            class_dir = os.path.join(img_dir, class_name)
            for img_name in os.listdir(class_dir):
                # Store the path to image & its numerical label
                self.images.append(
                    (os.path.join(class_dir, img_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Look up the address and label in our master list
        img_path, label = self.images[idx]

        # 2. Open the img file
        # We use .convert('RGB') to ensure img have 3 channels (Some may be grayscale but models usually expect 3 channels)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# --- THE LOADER FACTORY ---
def get_dataloaders(base_path, batch_size=32, seed=SEED):
    set_seed(seed)

    # Path Setup
    train_img_dir = os.path.join(base_path, 'train', 'images')
    val_img_dir = os.path.join(base_path, 'validation', 'images')

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5042, 0.5042, 0.5042], std=[
                             0.2058, 0.2058, 0.2058])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5042, 0.5042, 0.5042], std=[
                             0.2058, 0.2058, 0.2058])
    ])

    # Datasets
    train_data = NEUDataset(img_dir=train_img_dir, transform=train_transforms)
    val_data = NEUDataset(img_dir=val_img_dir, transform=val_transforms)

    # Generator for DataLoader
    g = torch.Generator()
    g.manual_seed(seed)

    # Loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=2, worker_init_fn=worker_init_fn, generator=g
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=2, worker_init_fn=worker_init_fn, generator=g
    )

    return train_loader, val_loader, train_data.classes


if __name__ == "__main__":
    # This code ONLY runs if you do: python dataset.py
    # Great for double-checking your paths work!
    print("Testing DataLoader setup...")
    train, val, classes = get_dataloaders(base_path='./NEU-DET')
    print(f"Success! Found classes: {classes}")

    # Check one batch
    images, labels = next(iter(train))
    print(f"Batch shape: {images.shape}")
