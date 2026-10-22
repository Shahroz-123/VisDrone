"""Fine-tune YOLOv8 on the VisDrone-DET dataset and validate the best checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("configs/visdrone.yaml"))
    parser.add_argument("--model", default="yolov8s.pt", help="Ultralytics model or checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="Device such as 0, 0,1, or cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="visdrone")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        parser.error("epochs, imgsz, and batch must be positive; workers cannot be negative")
    if not args.data.is_file():
        parser.error(f"Dataset configuration does not exist: {args.data}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        pretrained=True,
    )

    best_checkpoint = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training completed without a best checkpoint: {best_checkpoint}")

    validation = YOLO(str(best_checkpoint)).val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Validation mAP@0.5: {validation.box.map50:.4f}")
    print(f"Validation mAP@0.5:0.95: {validation.box.map:.4f}")


if __name__ == "__main__":
    main()
