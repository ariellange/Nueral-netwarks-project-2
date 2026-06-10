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