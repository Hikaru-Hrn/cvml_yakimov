import cv2
import torch
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Устройство: {device}")

model_path = "runs/detect/figures/yolo/weights/best.pt"

try:
    model = YOLO(model_path).to(device)
    print("Модель для распознавания фигур загружена.")
except Exception as e:
    print(f"Не удалось найти 'best.pt'. Ошибка: {e}")
    exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True, conf=0.6)

    for r in results:
        annotated_frame = r.plot()

        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = box.conf[0]
            print(f"Вижу объект: {label} ({confidence:.2f})")

    cv2.imshow("YOLO: Cubes and Spheres Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
