# src/task4_visualize.py
import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision import datasets

# Ensure plots folder exists
os.makedirs('../plots', exist_ok=True)

# Grab the raw dataset without normalization transforms to see clear colors
raw_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
sample_img, sample_label = raw_dataset[0]  # Grab a sample image
class_name = raw_dataset.classes[sample_label]

# Apply functional transformations individually to the exact same image
rotated = TF.rotate(sample_img, 25)
cropped = TF.resize(TF.crop(sample_img, top=4, left=4, height=24, width=24), (32, 32))
flipped = TF.hflip(sample_img)
# Crop tightly down to simulate a zoom, then resize back to square dimensions
zoomed   = TF.resize(TF.center_crop(sample_img, (22, 22)), (32, 32))

# Subplot rendering array
fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
titles = ['Original File', 'Rotation (25°)', 'Random Crop (+Pad)', 'Horizontal Flip', 'Zoom / Resize']
images = [sample_img, rotated, cropped, flipped, zoomed]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('../plots/task4_transform_proof.png', dpi=300)
print("Transformation visual baseline saved at: ../plots/task4_transform_proof.png")
plt.show()


# ---------------------------------------------------------------

# src/task4_augment_train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Import modules from our established infrastructure backend
from models import InitialCNN
from utils import train, test, test_loader, device, aug_train_loader

# Initialize a completely fresh instance of your baseline CNN starter network
aug_model = InitialCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(aug_model.parameters(), lr=0.001)

num_epochs = 15
aug_history = {'train_loss': [], 'test_acc': []}

print("\n--- Starting Training: InitialCNN with Augmented Dataset ---")
for epoch in range(num_epochs):
    # Pass variables explicitly to ensure local variable isolation
    train_loss, train_acc, _ = train(aug_model, aug_train_loader, optimizer, criterion)
    test_loss, test_acc = test(aug_model, test_loader, criterion)
    
    aug_history['train_loss'].append(train_loss)
    aug_history['test_acc'].append(test_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    print("-" * 40)

# Save your model weights locally so you never lose your progress!
torch.save(aug_model.state_dict(), 'aug_cnn_weights.pth')
print("Training complete. Augmented model weights secured successfully.")