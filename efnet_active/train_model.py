import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device}")


class Model:
    def __init__(self, weights, model_type="alexnet"):
        self.weights = weights
        self.model_type = model_type
        self.model = self.build_model()

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimazer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001
        )
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def build_model(self):
        if self.model_type == "alexnet":
            model = torchvision.models.alexnet(weights=self.weights)
            for param in model.features.parameters():
                param.requires_grad = False

            features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(features, 1)

        if self.model_type == "efficientnet":
            model = torchvision.models.efficientnet_b0(weights=self.weights)
            for param in model.features.parameters():
                param.requires_grad = False

            features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(features, 1)

        return model.to(device)

    def find_weights(self, path):
        model_path = path
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Успешно загружены веса из {model_path}")
        else:
            print("Файл модели не найден. Используются веса по умолчанию.")

    def train(self, buffer):
        if len(buffer) < 10:
            return None
        self.model.train()
        images, labels = buffer.get_batch()
        self.optimazer.zero_grad()
        predictions = self.model(images).squeeze(1)
        loss = self.criterion(predictions, labels)
        loss.backward()
        self.optimazer.step()
        return loss.item()

    def predict(self, frame):
        self.model.eval()
        tensor = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            predicted = self.model(tensor).squeeze()
            prob = torch.sigmoid(predicted).item()
        label = "person" if prob > 0.5 else "no_person"
        return label, prob


class Buffer:
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)

    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)

    def get_batch(self):
        images = torch.stack(list(self.frames)).to(device)
        labels = torch.tensor(list(self.labels), dtype=torch.float32).to(device)

        return images, labels


modelA = Model(torchvision.models.AlexNet_Weights.IMAGENET1K_V1, "alexnet")
modelB = Model(torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1, "efficientnet")

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
bufferA = Buffer()
bufferB = Buffer()
count_labeled = 0


while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if key == ord("q"):
        break

    elif key == ord("1"):  # person
        tensorA = modelA.transform(image)
        tensorB = modelB.transform(image)
        bufferA.append(tensorA, 1.0)
        bufferB.append(tensorB, 1.0)
        count_labeled += 1
        print(f"Labeled: {count_labeled}")

    elif key == ord("2"):  # no person
        tensorA = modelA.transform(image)
        tensorB = modelB.transform(image)
        bufferA.append(tensorA, 0.0)
        bufferB.append(tensorB, 0.0)
        count_labeled += 1
        print(f"Labeled: {count_labeled}")

    elif key == ord("p"):  # predict
        labelA, confidanceA = modelA.predict(frame)
        labelB, confidanceB = modelB.predict(frame)
        print(f"AlexNet: {labelA} ({confidanceA})")
        print(f"EficNet: {labelB} ({confidanceB})")

    elif key == ord("s"):  # save
        torch.save(model.state_dict(), "alexnet_custom.pth")
        print("Модель сохранена в alexnet_custom.pth")

    # print(len(buffer), count_labeled)
    if count_labeled >= bufferA.frames.maxlen:
        loss = train(bufferA)
        if loss:
            print(f"Loss = {loss}")
        count_labeled = 0
