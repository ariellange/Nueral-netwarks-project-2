# src/utils.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random

# Hardware Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed Settings for Reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Standard Transform (CIFAR-10 Normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Base Dataset Downloads
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Task 3 Train-Validation Split (45,000 / 5,000)
train_subset, val_subset = random_split(full_train_dataset, [45000, 5000])

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def train(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0, 0, 0
    gradient_norms = {}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms.setdefault(name, []).append(grad_norm)

        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100 * correct / total, gradient_norms

def test(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100 * correct / total


# ---------------------------------------------------------------

# 1. Dedicated Augmentation Transforms for Training Data
augmented_train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),                  # 1. Rotation
    transforms.RandomCrop(32, padding=4),                   # 2. Random Crop
    transforms.RandomHorizontalFlip(p=0.5),                 # 3. Horizontal Flip
    # RandomResizedCrop performs both cropping (Zoom) and Resizing back to 32x32
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),     # 4. Zoom/Resize
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 2. Re-instantiate the full training set with the augmentation pipeline
aug_full_train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=augmented_train_transform
)

# 3. Apply the exact same deterministic split using our global seed
aug_train_subset, _ = random_split(aug_full_train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(seed))

# 4. Final Augmented DataLoader for upcoming training loops
aug_train_loader = DataLoader(aug_train_subset, batch_size=128, shuffle=True)