# Phase 2 Day 4 - Transfer Learning with Pre-Trained ResNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert 1-channel to 3-channel
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Load pre-trained ResNet (trained on ImageNet with millions of images)
print("Loading pre-trained ResNet18...")
model = models.resnet18(pretrained=True)

# Freeze all layers except the last one
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer for our 10 digits
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Unfreeze the last layer for training
model.fc.requires_grad = True

model = model.to(device)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}\n")

# Train (only the last layer)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)

print("Training with transfer learning...")
start = time.time()

for epoch in range(3):
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}")

training_time = time.time() - start
print(f"Training time: {training_time:.1f}s\n")

# Test accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"Transfer Learning Test Accuracy: {accuracy:.2f}%")

# Compare all three approaches
print("\n=== Comparison ===")
print("Fully Connected NN:  96.25% (trained 3 epochs)")
print("CNN from scratch:    99.19% (trained 5 epochs, 161s)")
print(f"Transfer Learning:   {accuracy:.2f}% (trained 3 epochs, {training_time:.1f}s)")