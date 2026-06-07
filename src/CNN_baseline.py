# src/task4_baseline_train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Import our backend infrastructure variables
from models import InitialCNN
from utils import train, test, train_loader, val_loader, device

# Initialize your standard starter network
baseline_cnn = InitialCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_cnn.parameters(), lr=0.001)

num_epochs = 15
baseline_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print("\n--- Starting Training: InitialCNN WITHOUT Augmentation (Baseline) ---")
for epoch in range(num_epochs):
    train_loss, train_acc, _ = train(baseline_cnn, train_loader, optimizer, criterion)
    val_loss, val_acc = test(baseline_cnn, val_loader, criterion)
    
    baseline_history['train_loss'].append(train_loss)
    baseline_history['val_loss'].append(val_loss)
    baseline_history['train_acc'].append(train_acc)
    baseline_history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
    print("-" * 40)

# Save the trained baseline weights to your local workspace folder
torch.save(baseline_cnn.state_dict(), 'baseline_cnn_weights.pth')
print("Baseline model training complete. Weights secured as 'baseline_cnn_weights.pth'.")