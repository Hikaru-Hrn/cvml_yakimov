import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from train_model import CyrillicCnn, ZipDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    try:
        dataset = ZipDataset("cyrillic.zip", transform=transform)
        idx2class = {v: k for k, v in dataset.class2idx.items()}
    except Exception as e:
        print(f"Ошибка загрузки архива: {e}")
        return

    model = CyrillicCnn(dataset.num_classes).to(device)
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.eval()
        print("Модель успешно загружена!")
    except FileNotFoundError:
        print("Файл model.pth не найден. Сначала запусти train_model.py")
        return

    test_img, test_label = dataset[10000]
    input_tensor = test_img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()

    print(f"Результат теста:")
    print(f"Реальная буква: {idx2class[test_label]}")
    print(f"Предсказание модели: {idx2class[prediction]}")

    if test_label == prediction:
        print("Модель угадала символ.")
    else:
        print("Модель ошиблась.")


if __name__ == "__main__":
    main()
