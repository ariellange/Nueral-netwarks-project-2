import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os

# Ensure plots folder exists
os.makedirs('../plots', exist_ok=True)

# ====================================================================
# 1. Data Loading (With Task 4 Augmentation for Training Only)
# ====================================================================
def get_augmented_data_loaders(batch_size=64):
    print("--- Loading CIFAR-10 Data with Task 4 Augmentations ---")
    
    # Training Data gets the full augmentation treatment
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Test Data MUST stay pristine (only normalized)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# ====================================================================
# 2. Model Definition (Directly using your partner's ImprovedCNN)
# ====================================================================
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ====================================================================
# 3. Training & Testing Functions (Identical to your template)
# ====================================================================
def train_model(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy

def test_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy

# ====================================================================
# 4. Confusion Matrix Generation
# ====================================================================
def generate_confusion_matrix(model, loader, device, title, filename):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'../plots/{filename}', dpi=300)
    print(f"Confusion Matrix saved to: ../plots/{filename}")
    plt.show()

# ====================================================================
# 5. Execution Block
# ====================================================================
if __name__ == '__main__':
    # Determine execution device dynamically
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on hardware: {device}")

    train_loader, test_loader = get_augmented_data_loaders(batch_size=64)
    
    # Initialize the model blueprint
    cnn_model = ImprovedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    num_epochs = 15
    print(f"\n--- Training ImprovedCNN WITH Augmentation for {num_epochs} Epochs ---")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(cnn_model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test_model(cnn_model, test_loader, criterion, device)
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}%")

    # Save the weights safely
    torch.save(cnn_model.state_dict(), 'improved_augmented_cnn.pth')
    
    # Generate the professional Task 4 Confusion Matrix
    generate_confusion_matrix(
        cnn_model, 
        test_loader, 
        device, 
        title='Improved CNN Test Performance WITH Augmentation', 
        filename='task4_augmented_confusion_matrix.png'
    )