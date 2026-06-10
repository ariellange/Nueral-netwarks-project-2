import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import urllib.request

# Disable the download progress bar printouts from urllib
urllib.request.urlcleanup()

# =====================================================================
# GLOBAL CONFIGURATION
# =====================================================================
NUM_EPOCHS = 15  # Allowed max epochs, but Early Stopping will cut it short!
SEED = 42
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 3  # Stop training if test loss doesn't improve for 3 epochs

# =====================================================================
# 1. Reproducibility Settings (Seed Initialization)
# =====================================================================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =====================================================================
# 2. CIFAR-10 Dataset Loading and Preprocessing
# =====================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading CIFAR-10 dataset...")
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Dataset loaded successfully.")

# =====================================================================
# 3. Hybrid Network Architecture Definition (HybridNet)
# =====================================================================
class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()
        
        # Part 1: CNN Feature Extractor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spatial_dropout = nn.Dropout2d(p=0.1)
        
        # Part 2: MLP Classifier
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout_fc = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.pool(x)
        x = self.spatial_dropout(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        x = self.fc2(x)
        return x

model = HybridNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Dynamic learning rate reduction when learning plateaus (verbose removed for PyTorch compatibility)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

# =====================================================================
# 4. Training and Evaluation Core Functions
# =====================================================================
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(loader.dataset), correct / total

def test(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader.dataset), correct / total

# =====================================================================
# 5. Main Optimization Loop with Early Stopping & LR Scheduling
# =====================================================================
train_losses, test_losses = [], []
train_accs, test_accs = [], []

best_test_loss = float('inf')
patience_counter = 0
actual_epochs_run = 0

print("\n--- Starting Training Stage ---")
for epoch in range(NUM_EPOCHS):
    actual_epochs_run += 1
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    test_loss, test_acc = test(model, test_loader, criterion)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc * 100)
    test_accs.append(test_acc * 100)
    
    print(f"Epoch {epoch+1:02d} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc * 100:.2f}%")
    
    # Step the learning rate scheduler based on current test loss performance
    scheduler.step(test_loss)
    
    # Early Stopping Guard Mechanism
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0  # Reset counter since we found a better state
    else:
        patience_counter += 1
        print(f"-> No improvement in test loss. Early stopping counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n[Early Stopping Triggered] Training halted automatically at epoch {epoch+1} to avoid overfitting.")
        break

print("\n--- Training Process Completed ---")

# =====================================================================
# 6. Performance Visualizations (Dynamic X-Axis based on run duration)
# =====================================================================
epochs_range = range(1, actual_epochs_run + 1)

# Figure 1: Loss curves
plt.figure(1)
plt.plot(epochs_range, train_losses, label='Train Loss', color='blue', marker='o')
plt.plot(epochs_range, test_losses, label='Test Loss', color='red', marker='x')
plt.title('HybridNet - Learning Curves: Categorical Loss (with Early Stopping)')
plt.xticks(epochs_range)
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend()
plt.grid(True)

# Figure 2: Accuracy curves
plt.figure(2)
plt.plot(epochs_range, train_accs, label='Train Accuracy', color='blue', marker='o')
plt.plot(epochs_range, test_accs, label='Test Accuracy', color='red', marker='x')
plt.title('HybridNet - Learning Curves: Accuracy Score (%)')
plt.xticks(epochs_range)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100) 
plt.legend()
plt.grid(True)

plt.show()