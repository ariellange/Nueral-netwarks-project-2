import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Importing from neighboring files
from models import SmallMLP, LargeMLP
from utils import train, test, train_loader, val_loader, device

# Ensure plots folder exists outside src
os.makedirs('../plots', exist_ok=True)

num_epochs = 15
criterion = nn.CrossEntropyLoss()


experiments = {
    'Small MLP (Under/Well-Fitted)': {
        'model': SmallMLP().to(device), 
        'opt': lambda m: optim.Adam(m.parameters(), lr=0.001)
    },
    'Large MLP (Overfitted Baseline)': {
        'model': LargeMLP().to(device), 
        'opt': lambda m: optim.Adam(m.parameters(), lr=0.001)
    }
}

histories = {}

# Execution Block running modules sequentially
for name, config in experiments.items():
    print(f"\n--- Training Model: {name} ---")
    model = config['model']
    optimizer = config['opt'](model)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_acc, _ = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = test(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {val_loss:.4f} | Test Accuracy: {val_acc:.2f}%")
        print("-" * 40)
        
    histories[name] = history

small_history = histories['Small MLP (Under/Well-Fitted)']
large_history = histories['Large MLP (Overfitted Baseline)']

# ====================================================================
# VISUALIZATION GENERATION
# ====================================================================
epochs = range(1, num_epochs + 1)

# Graph 1: Standard Epoch Curves (Required by prompt)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(epochs, large_history['train_loss'], 'b-o', label='Train Loss')
ax1.plot(epochs, large_history['val_loss'], 'r-o', label='Validation Loss')
ax1.set_title('LargeMLP Model: Loss vs. Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle=':')
ax1.legend()

ax2.plot(epochs, large_history['train_acc'], 'b-x', label='Train Accuracy')
ax2.plot(epochs, large_history['val_acc'], 'r-x', label='Validation Accuracy')
ax2.set_title('LargeMLP Model: Accuracy vs. Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle=':')
ax2.legend()
plt.tight_layout()
plt.savefig('../plots/task3_LargeMLP_curves.png', dpi=300)
plt.close()

# Graph 2: Custom Velocity/Gap Chart (Used to pinpoint Epoch 6)
baseline_train_acc = np.array(large_history['train_acc'])
baseline_val_acc = np.array(large_history['val_acc'])
gen_gap = baseline_train_acc - baseline_val_acc
val_acc_change = np.diff(baseline_val_acc)

fig, ax1 = plt.subplots(figsize=(11, 6))
color = 'tab:purple'
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Validation Accuracy Change (Percentage Points)', color=color, fontsize=12)
ax1.bar(range(2, 16), val_acc_change, color=color, alpha=0.6, label='Val Acc Change')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Generalization Gap (Train - Val %)', color=color, fontsize=12)
ax2.plot(range(1, 16), gen_gap, color=color, linestyle='--', marker='x', linewidth=2, label='Generalization Gap')
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(y=10, color='red', linestyle=':', linewidth=1.5, label='10% Overfit Threshold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.title('Large MLP Overfitting: Generalization Gap vs. Velocity of Val Accuracy', fontsize=14, pad=15)
fig.tight_layout()
plt.savefig('../plots/task3_overfitting_velocity.png', dpi=300)
plt.close()

# Update Graph 3 in src/task3.py to show the size comparison:
small_history = histories['Small MLP (Under/Well-Fitted)']
large_history = histories['Large MLP (Overfitted Baseline)']

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, small_history['val_loss'], 'g-', label='Small MLP (Stable Loss)', linewidth=2)
plt.plot(epochs, large_history['val_loss'], 'r--', label='Large MLP (Exploding Loss)', linewidth=2)
plt.title('Validation Loss: Model Size Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle=':')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, small_history['val_acc'], 'g-', label='Small MLP (Stable Acc)', linewidth=2)
plt.plot(epochs, large_history['val_acc'], 'r--', label='Large MLP (Dropping Acc)', linewidth=2)
plt.title('Validation Accuracy: Model Size Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle=':')
plt.legend()

plt.tight_layout()
plt.savefig('../plots/task3_size_comparison.png', dpi=300)
plt.close()

print("\nTask 3 pipeline complete. All three report graphics saved inside the /plots/ directory.")