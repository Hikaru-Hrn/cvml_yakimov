import io
import time
from pathlib import PurePosixPath
from zipfile import ZipFile

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")


class ZipDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.archive = None
        self.transform = transform
        self.file_paths = []
        self.labels = []

        with ZipFile(self.path, "r") as archive:
            all_names = archive.namelist()

            folder_names = set()
            for name in all_names:
                path = PurePosixPath(name)
                if len(path.parts) >= 2:
                    folder_name = path.parts[-2]
                    if folder_name != "Cyrillic":
                        folder_names.add(folder_name)

            folder_names = sorted(list(folder_names))
            self.class2idx = {name: i for i, name in enumerate(folder_names)}
            self.num_classes = len(folder_names)

            for name in all_names:
                if name.lower().endswith((".png")):
                    letter = name.split("/")[-2]
                    self.file_paths.append(name)
                    self.labels.append(self.class2idx[letter])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.archive is None:
            self.archive = ZipFile(self.path, "r")

        img_name = self.file_paths[idx]
        with self.archive.open(img_name) as f:
            image = Image.open(f).split()[-1]
            label = self.labels[idx]

            return self.transform(image), label


class CyrillicCnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.poll = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.poll(F.relu(self.bn1(self.conv1(x))))
        x = self.poll(F.relu(self.bn2(self.conv2(x))))

        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = self.dropout2(x)
        x = self.fc2(x)
        return x


def learn():
    batch_size = 64
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = ZipDataset("cyrillic.zip")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_set.dataset.transform = transform
    test_set.dataset.transform = transform

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = CyrillicCnn(dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20

    history = {"loss": [], "acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == lbls).sum().item()

        acc = 100 * correct / len(train_set)
        history["loss"].append(total_loss / len(train_loader))
        history["acc"].append(acc)

        model.eval()
        test_correct = 0
        with torch.no_grad():
            for t_imgs, t_lbls in test_loader:
                t_imgs, t_lbls = t_imgs.to(device), t_lbls.to(device)
                t_out = model(t_imgs)
                test_correct += (t_out.argmax(1) == t_lbls).sum().item()

        test_acc = 100 * test_correct / len(test_set)