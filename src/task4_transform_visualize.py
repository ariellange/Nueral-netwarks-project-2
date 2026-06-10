import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Create plots folder if it doesn't exist
os.makedirs('../plots', exist_ok=True)

# 1. Load the dataset with a completely raw transform (just to get an image)
raw_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
# Get a clean image (Index 7 is a horse, index 4 is an automobile)
original_tensor, label = raw_dataset[4]

# Convert tensor to PIL Image to apply individual transforms cleanly
to_pil = transforms.ToPILImage()
original_img = to_pil(original_tensor)

# 2. Define Task 4 transformations individually
transform_rotation = transforms.RandomRotation(degrees=(15, 15)) # Force exactly 15 deg for clear demo
transform_crop     = transforms.RandomCrop(32, padding=4)
transform_flip     = transforms.RandomHorizontalFlip(p=1.0)       # Force flip for demo
transform_resize   = transforms.RandomResizedCrop(32, scale=(0.8, 0.8))

# 3. Apply transformations
img_rotated = transform_rotation(original_img)
img_cropped = transform_crop(original_img)
img_flipped = transform_flip(original_img)
img_resized = transform_resize(original_img)

# 4. Set up the Side-by-Side Plot
fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
images = [original_img, img_rotated, img_cropped, img_flipped, img_resized]
titles = [
    "Original Image", 
    "Random Rotation\n(±15°)", 
    "Random Crop\n(32x32, pad=4)", 
    "Horizontal Flip\n(p=0.5)", 
    "Random Resized Crop\n(Scale 0.8-1.0)"
]

for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.set_title(titles[i], fontsize=10, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('../plots/task4_transform_proof.png', dpi=300)
print("Transformation proof plot successfully saved to: ../plots/task4_transform_proof.png")
plt.show()


# 1. Load your existing full results files (handling device mapping safely)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Adjust these filenames if they are named slightly differently in your directory
baseline_results = torch.load('improved_baseline_cnn.pth', map_location=device)
augmented_results = torch.load('src/improved_augmented_cnn.pth', map_location=device)

# 2. Extract the 15-epoch accuracy history arrays directly from your files
# Note: If your saving script used keys like 'train_acc'/'test_acc' instead, change these string keys
train_unaug = baseline_results['train_acc']
test_unaug  = baseline_results['test_acc']

train_aug   = augmented_results['train_acc']
test_aug    = augmented_results['test_acc']

epochs = range(1, len(train_unaug) + 1)

# 3. Construct the 15-Epoch Comparison Chart
plt.figure(figsize=(11, 7))

# Plot WITHOUT Augmentation (Orange/Red to show the overfitting gap)
plt.plot(epochs, train_unaug, color='tab:orange', linestyle='-', linewidth=2.5, marker='o', label='Train Accuracy (Un-augmented)')
plt.plot(epochs, test_unaug, color='tab:red', linestyle='--', linewidth=1.5, marker='x', label='Test Accuracy (Un-augmented)')

# Plot WITH Augmentation (Blue/Cyan to show the stabilized generalization)
plt.plot(epochs, train_aug, color='tab:blue', linestyle='-', linewidth=2.5, marker='s', label='Train Accuracy (Augmented)')
plt.plot(epochs, test_aug, color='tab:cyan', linestyle='-', linewidth=2, marker='D', label='Test Accuracy (Augmented)')

# 4. Formatting and Metadata
plt.title('Improved CNN (ReLU+BN): 15-Epoch Augmentation Impact on Generalization', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Training Horizon (Epochs)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(epochs)
plt.ylim(30, 100)

plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True, title='Configuration Profiles')
plt.tight_layout()

# 5. Save the output
plt.savefig('../plots/task4_epoch_accuracy_comparison.png', dpi=300)
print("Success! 15-epoch accuracy comparison plot saved to: ../plots/task4_epoch_accuracy_comparison.png")
plt.show()