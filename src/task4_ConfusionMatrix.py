# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import os

# # ====================================================================
# # 1. Standard Data Loading (No Augmentation)
# # ====================================================================
# def get_baseline_data_loaders(batch_size=64):
#     print("--- Loading Pristine CIFAR-10 Data (No Augmentation) ---")
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

#     test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader

# # ====================================================================
# # 2. Model Definition
# # ====================================================================
# class ImprovedCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#         self.fc1 = nn.Linear(256 * 2 * 2, 256)
#         self.fc2 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv4(x))
#         x = F.max_pool2d(x, 2)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # ====================================================================
# # 3. Training & Testing Loops
# # ====================================================================
# def train_model(model, loader, optimizer, criterion, device):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for images, labels in loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
#     return running_loss / len(loader), 100 * correct / total

# def test_model(model, loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#     return running_loss / len(loader), 100 * correct / total

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     train_loader, test_loader = get_baseline_data_loaders(batch_size=64)
    
#     model = ImprovedCNN().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     print("\n--- Training ImprovedCNN WITHOUT Augmentation ---")
#     for epoch in range(15):
#         train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
#         test_loss, test_acc = test_model(model, test_loader, criterion, device)
#         print(f"Epoch {epoch+1:02d}/15 | Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}%")

#     torch.save(model.state_dict(), 'improved_baseline_cnn.pth')
#     print("Baseline model weights saved as 'improved_baseline_cnn.pth'")



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


# Re-declare model structure for loading state dicts
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

def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('../plots', exist_ok=True)

    # Standard clean test dataset for evaluating both models
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 1. Load Baseline Weights
    baseline_model = ImprovedCNN().to(device)
    baseline_model.load_state_dict(torch.load('improved_baseline_cnn.pth', map_location=device))
    y_true_base, y_pred_base = get_predictions(baseline_model, test_loader, device)
    cm_base = confusion_matrix(y_true_base, y_pred_base)

    # 2. Load Augmented Weights
    augmented_model = ImprovedCNN().to(device)
    augmented_model.load_state_dict(torch.load('src/improved_augmented_cnn.pth', map_location=device))
    y_true_aug, y_pred_aug = get_predictions(augmented_model, test_loader, device)
    cm_aug = confusion_matrix(y_true_aug, y_pred_aug)

    # 3. Plot Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Left Panel: Baseline
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[0].set_title('CNN Test Performance WITHOUT Augmentation', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)

    # Right Panel: Augmented
    sns.heatmap(cm_aug, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[1].set_title('CNN Test Performance WITH Augmentation', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.savefig('../plots/task4_comparison_matrices.png', dpi=300)
    print("Master comparison chart saved to: ../plots/task4_comparison_matrices.png")
    plt.show()