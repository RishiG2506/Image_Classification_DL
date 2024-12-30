import PIL
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = "cpu"


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)
        self.transform = transform
        self.features = np.load(f"{img_dir.split('_')[0]}_features.npy")
        self.features = torch.tensor(self.features, dtype=torch.float32, device=device)
        self.images = [
            PIL.Image.open(os.path.join(img_dir, img)).convert("RGB")
            for img in self.img_files
        ]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        label = int(self.img_files[idx].split("_")[0]) - 1
        # feature = self.features[idx]
        # label = int(feature[-1]) - 1
        # feature = feature[:-1]

        return image, label


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.dropout1(out)
        out = self.relu2(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


class Model:
    def __init__(self, resize_to=32):
        self.resized_dim = resize_to * resize_to * 3
        self.model = FeedForwardNN(
            input_dim=3072,
            hidden_dim_1=768,
            hidden_dim_2=768,
            output_dim=60,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((resize_to, resize_to)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((resize_to, resize_to)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def train(self, img_dir: str, epochs=35):
        train_dataset = CustomImageDataset(
            img_dir=img_dir, transform=self.train_transform
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            losses.append(0.0)
            for i, (images, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()

                images = images.view(labels.size(0), -1)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()
                running_loss += loss.item()
                losses[-1] += loss.item() / len(train_loader)
                if i % 10 == 9:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}"
                    )
                    running_loss = 0.0
        # plot
        import matplotlib.pyplot as plt

        plt.plot(losses)
        plt.show()

    def evaluate(self, train_img_dir, test_img_dir: str):
        train_dataset = CustomImageDataset(
            img_dir=train_img_dir, transform=self.test_transform
        )
        test_dataset = CustomImageDataset(
            img_dir=test_img_dir, transform=self.test_transform
        )
        for dataset in (train_dataset, test_dataset):
            test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.view(labels.size(0), -1)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Accuracy of the model on the test images: {100 * correct / total}%")


if __name__ == "__main__":
    model = Model()
    model.model.to(device)
    model.train("train_data")
    model.evaluate("train_data", "val_data")