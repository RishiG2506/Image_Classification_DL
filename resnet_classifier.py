import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import pywt
from torchvision import transforms, datasets
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = int(self.image_files[idx].split("_")[0]) - 1
        return image, label

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=60):
        super(ResNetClassifier, self).__init__()
        # Load a pre-trained ResNet model
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Freeze the feature extraction layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Hyperparameters
num_classes = 60
batch_size = 32
learning_rate = 1e-3
num_epochs = 10
device = torch.device("mps" if torch.mps.is_available() else "cpu")

# Data transformations (adjust if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
]) 

train_dataset = ImageDataset(img_dir='train_data', transform=transform)
test_dataset = ImageDataset(img_dir='val_data', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model setup
model = ResNetClassifier(num_classes=num_classes).to(device)

# Unfreeze all layers for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_model():
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Testing loop
def test_model():
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing Acc"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for inputs, labels in tqdm(train_loader, desc="Training Acc"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
    return 100 * correct_test / total_test, 100 * correct_train / total_train


max_test_accuracy = 0
# Training process
for epoch in range(num_epochs):
    train_loss = train_model()
    test_accuracy, train_accuracy = test_model()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy Test: {test_accuracy:.2f}%, Accuracy Train: {train_accuracy:.2f}%")
    if test_accuracy > max_test_accuracy:
        # Save the model
        torch.save(model.state_dict(), 'resnet_classifier.pth')
        max_test_accuracy = test_accuracy
        print("Model saved as resnet_classifier.pth")


