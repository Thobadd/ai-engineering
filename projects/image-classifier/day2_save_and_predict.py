# Phase 2 Day 2 - Save and Load Neural Networks
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same network as before
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Create and train model
print("Training model...")
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

# Save the model
print("\nSaving model...")
torch.save(model.state_dict(), 'model.pth')
print("Model saved as model.pth")

# Load the model
print("Loading model...")
loaded_model = SimpleNet().to(device)
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()
print("Model loaded successfully")

# Test on some random examples
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

print("\n=== Predictions on Real Test Images ===")
loaded_model.eval()
with torch.no_grad():
    for i, (image, true_label) in enumerate(test_loader):
        if i >= 5:  # Show 5 examples
            break
        
        image = image.to(device)
        output = loaded_model(image)
        predicted = output.argmax(1).item()
        confidence = torch.softmax(output, 1)[0][predicted].item()
        
        status = "✓" if predicted == true_label.item() else "✗"
        print(f"{status} Predicted: {predicted}, True: {true_label.item()}, Confidence: {confidence:.2%}")