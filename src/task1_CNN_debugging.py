import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ==========================================
# 1. Data Loading
# ==========================================
def get_data_loaders(batch_size=64):
    print("--- Loading CIFAR-10 Data ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# ==========================================
# 2. Models Definition
# ==========================================

# ---> THE IMPROVED CNN (ReLU + BatchNorm2d) <---
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

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 3. Training & Testing Functions
# ==========================================
def train_model(model, loader, optimizer, criterion):
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

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy

def test_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy

# ==========================================
# 4. Visualization Functions
# ==========================================
def plot_accuracies(cnn_train, cnn_test, mlp_train, mlp_test, num_epochs):
    print("--- Generating Accuracy Comparison Chart ---")
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cnn_train, 'b-', marker='o', label='CNN Train Acc')
    plt.plot(epochs, cnn_test, 'b--', marker='s', label='CNN Test Acc')
    plt.plot(epochs, mlp_train, 'r-', marker='o', label='MLP Train Acc')
    plt.plot(epochs, mlp_test, 'r--', marker='s', label='MLP Test Acc')

    plt.title('Improved CNN (ReLU+BN) vs. MLP: Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.show()

def visualize_weights(trained_cnn_model, trained_mlp_model):
    print("--- Generating Visualizations for Trained Weights ---")

    # CNN Weights
    cnn_weights = trained_cnn_model.conv1.weight.data.clone()
    fig_cnn, axes_cnn = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes_cnn.flat):
        weight = cnn_weights[i].permute(1, 2, 0).numpy()
        weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
        ax.imshow(weight)
        ax.axis('off')
    plt.suptitle("Trained CNN 'conv1' Filters (Edge/Color Detectors)")
    plt.show()

    # MLP Weights
    mlp_weights = trained_mlp_model.fc1.weight.data.clone()
    fig_mlp, axes_mlp = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes_mlp.flat):
        weight = mlp_weights[i].view(3, 32, 32).permute(1, 2, 0).numpy()
        weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
        ax.imshow(weight)
        ax.set_title(f'Neuron {i+1}')
        ax.axis('off')
    plt.suptitle("Trained MLP 'fc1' Weights (Global Noise/Blobs)")
    plt.show()

# ==========================================
# 5. Main Execution Block
# ==========================================
if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders(batch_size=64)

    mlp_model = MLP()
    cnn_model = ImprovedCNN() # ---> Using the Improved CNN here <---
    criterion = nn.CrossEntropyLoss()

    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001)
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)

    num_epochs = 15 # ---> Set to 15 for proper comparison <---

    cnn_train_accs, cnn_test_accs = [], []
    mlp_train_accs, mlp_test_accs = [], []

    print(f"\n--- Starting CNN Training & Evaluation for {num_epochs} Epochs ---")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(cnn_model, train_loader, optimizer_cnn, criterion)
        test_loss, test_acc = test_model(cnn_model, test_loader, criterion)

        cnn_train_accs.append(train_acc)
        cnn_test_accs.append(test_acc)

        print(f"CNN Epoch {epoch+1:02d}/{num_epochs} | Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}%")

    print(f"\n--- Starting MLP Training & Evaluation for {num_epochs} Epochs ---")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(mlp_model, train_loader, optimizer_mlp, criterion)
        test_loss, test_acc = test_model(mlp_model, test_loader, criterion)

        mlp_train_accs.append(train_acc)
        mlp_test_accs.append(test_acc)

        print(f"MLP Epoch {epoch+1:02d}/{num_epochs} | Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}%")

    # Execute Visualizations
    plot_accuracies(cnn_train_accs, cnn_test_accs, mlp_train_accs, mlp_test_accs, num_epochs)
    visualize_weights(cnn_model, mlp_model)