import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import pywt

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, use_pca=False, use_wavelet=False):
        self.img_dir = img_dir
        self.transform = transform
        self.use_pca = use_pca
        self.use_wavelet = use_wavelet
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images to 128x128
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        
        # Get image files
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        # Initialize PCA if needed
        if use_pca:
            self.pca = PCA(n_components=32)
            self.fit_pca()

    def fit_pca(self):
        # Collect sample of images to fit PCA
        sample_images = []
        for img_file in self.image_files[:500]:  # Use subset for efficiency
            img_path = os.path.join(self.img_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).reshape(-1, 3)
            sample_images.append(img)
        sample_images = np.vstack(sample_images)
        self.pca.fit(sample_images)

    def apply_wavelet(self, img):
        # Apply wavelet transform to each channel
        coeffs = []
        for channel in range(3):
            coeffs_channel = pywt.dwt2(img[:,:,channel], 'haar')
            coeffs.append(coeffs_channel)
        
        # Reconstruct image using approximation coefficients
        transformed = np.stack([c[0] for c in coeffs], axis=2)
        return transformed

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.use_wavelet:
            image = np.array(image)
            image = self.apply_wavelet(image)
            image = Image.fromarray(image.astype('uint8'))
        
        if self.transform:
            image = self.transform(image)
            
        if self.use_pca:
            # Reshape and apply PCA
            img_flat = image.numpy().reshape(-1, 3)
            img_pca = self.pca.transform(img_flat)
            image = torch.FloatTensor(img_pca.reshape(32, -1))

        label = int(self.image_files[idx].split("_")[0]) - 1
        return image, label

class CNNModel(nn.Module):
    def __init__(self, num_classes=60):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 2, 128),  # Adjust based on your input image size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=30):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # Print metrics
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        # Learning rate scheduling
        # scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'cnn_classifier.pth')

def main():
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(
        img_dir='train_data',
        transform=transform,
        use_pca=False,  # Set to True to use PCA
        use_wavelet=True  # Set to True to use wavelet transform
    )
    
    test_dataset = ImageDataset(
        img_dir='val_data',
        transform=transform,
        use_pca=False,
        use_wavelet=True
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize and train model
    model = CNNModel(num_classes=60)
    train_model(model, train_loader, test_loader)

if __name__ == "__main__":
    main()