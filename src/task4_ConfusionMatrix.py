# src/task4_confusion_matrix.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import back-end utilities
from models import InitialCNN
from utils import test_loader, device, transform

# Ensure plots folder exists
os.makedirs('../plots', exist_ok=True)

# 1. Load trained Baseline CNN weights
baseline_model = InitialCNN().to(device)
if os.path.exists('baseline_cnn_weights.pth'):
    baseline_model.load_state_dict(torch.load('baseline_cnn_weights.pth', map_location=device))
else:
    print("WARNING: 'baseline_cnn_weights.pth' not found. Running random weights!")
baseline_model.eval()

# 2. Load trained Augmented CNN weights
aug_model = InitialCNN().to(device)
if os.path.exists('aug_cnn_weights.pth'):
    aug_model.load_state_dict(torch.load('aug_cnn_weights.pth', map_location=device))
else:
    print("WARNING: 'aug_cnn_weights.pth' not found. Running random weights!")
aug_model.eval()

def extract_predictions(model, loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

print("Extracting predictions from both models...")
labels_true, baseline_preds = extract_predictions(baseline_model, test_loader)
_, aug_preds = extract_predictions(aug_model, test_loader)

# Core CIFAR-10 category labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Compute numerical matrices
cm_baseline = confusion_matrix(labels_true, baseline_preds)
cm_aug = confusion_matrix(labels_true, aug_preds)

# ====================================================================
# RENDERING COMPARATIVE GRAPH
# ====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Panel A: Baseline Matrix
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Reds', ax=ax1,
            xticklabels=class_names, yticklabels=class_names, cbar=False)
ax1.set_title('CNN Test Performance WITHOUT Augmentation', fontsize=12, fontweight='bold')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

# Panel B: Augmented Matrix
sns.heatmap(cm_aug, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=class_names, yticklabels=class_names, cbar=False)
ax2.set_title('CNN Test Performance WITH Augmentation', fontsize=12, fontweight='bold')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

plt.tight_layout()
plt.savefig('../plots/task4_confusion_matrices.png', dpi=300)
print("Comparative matrix plot compiled and secured at: ../plots/task4_confusion_matrices.png")
plt.show()