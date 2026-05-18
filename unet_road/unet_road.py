from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

path = Path("../../resources/roads")
model_path = Path("unet_road.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoadsDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.images_path = path / "images"
        self.masks_path = path / "masks"
        self.images = list(self.images_path.glob("*.png"))
        self.masks = list(self.masks_path.glob("*.png"))
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image = image.resize((256, 256))
        image = np.array(image) / 255
        mask = Image.open(self.masks[index]).convert("L")
        mask = mask.resize((256, 256))
        mask = np.array(mask, dtype="f4")
        mask = (mask == 82).astype("f4")
        mask = np.expand_dims(mask, axis=0)
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=2).copy()
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask)
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downscale = nn.ModuleList()
        self.upscale = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for n in features:
            self.downscale.append(DoubleConv(in_channels, n))
            in_channels = n

        for n in reversed(features):
            self.upscale.append(nn.ConvTranspose2d(n * 2, n, 2, 2))
            self.upscale.append(DoubleConv(n * 2, n))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.result = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []

        for ds in self.downscale:
            x = ds(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx in range(0, len(self.upscale), 2):
            x = self.upscale[idx](x)
            skip = skips[idx // 2]
            cx = torch.cat((skip, x), dim=1)
            x = self.upscale[idx + 1](cx)
        return self.result(x)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        p_area = pred_sig.view(-1)
        t_area = target.view(-1)
        intersection = (p_area * t_area).sum()
        return 1 - (2 * intersection + 1) / (p_area.sum() + t_area.sum() + 1)


ds = RoadsDataset(path)

model = UNet().to(device)

criterion = DiceLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
images_loader = DataLoader(ds, batch_size=4)

print(f"Найдено изображений: {len(ds.images)}")
print(f"Найдено масок: {len(ds.masks)}")

if len(ds) == 0:
    raise ValueError(f"Ошибка: Не найдено изображений по пути {path.resolve()}")

num_epochs = 20
train_loss = []
train_acc = []

if not model_path.exists():
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        total = 0
        correct = 0
        for idx, (images, masks) in enumerate(images_loader):
            images, masks = (images.to(device), masks.to(device))
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            # total += masks.size(0)
            # correct += (predicted == masks).sum().item()
            with torch.no_grad():
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == masks).sum().item()
                total += masks.numel()
        scheduler.step()
        epoch_loss = run_loss / len(images_loader)
        epoch_acc = 100 * (correct / total)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f"Epoch {epoch}: epoch loss = {epoch_loss:=.3f}, acc = {epoch_acc:=.3f}")
    torch.save(model.state_dict(), model_path)
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
