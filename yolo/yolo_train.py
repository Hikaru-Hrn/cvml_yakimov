from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def train_model():
    classes = {0: "cube", 1: "neither", 2: "sphere"}
    root = Path(__file__).parent / "dataset"

    cfg = {
        "path": str(root.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }

    yaml_path = root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    size = "s"
    model = YOLO(f"yolo26{size}.pt")

    result = model.train(
        data=str(yaml_path.absolute()),
        imgsz=640,
        batch=8,
        workers=6,
        epochs=50,
        patience=5,
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=3,
        cos_lr=True,
        dropout=0.25,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        degrees=5.0,
        scale=0.5,
        translate=0.1,
        conf=0.001,
        iou=0.7,
        project="figures",
        name="yolo",
        save=True,
        save_period=5,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=True,
        plots=True,
        val=True,
        close_mosaic=8,
        amp=True,
    )

    print("Готово")
    print(f"Результат сохранен: {result.save_dir}")


if __name__ == "__main__":
    train_model()
