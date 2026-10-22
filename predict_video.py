"""Run YOLO inference on a video and save annotated frames as an MP4."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Input video file")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/visdrone/weights/best.pt"),
        help="Trained YOLO checkpoint",
    )
    parser.add_argument("--output", type=Path, help="Output MP4 path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None, help="Device such as 0 or cpu")
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"Input video does not exist: {args.source}")
    if not args.weights.is_file():
        parser.error(f"Model checkpoint does not exist: {args.weights}")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("conf must be between 0 and 1")
    if args.imgsz <= 0:
        parser.error("imgsz must be positive")

    output_path = args.output or args.source.with_name(
        f"{args.source.stem}_detections.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise OSError(f"Could not open input video: {args.source}")

    writer = None
    frames_processed = 0
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        output_fps = source_fps if source_fps > 0 else 30.0
        started_at = time.perf_counter()

        while True:
            success, frame = capture.read()
            if not success:
                break

            result = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            annotated = result.plot()
            frames_processed += 1
            elapsed = time.perf_counter() - started_at
            measured_fps = frames_processed / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                annotated,
                f"Processing FPS: {measured_fps:.1f}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise OSError(f"Could not create output video: {output_path}")
            writer.write(annotated)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frames_processed == 0:
        raise ValueError(f"No frames could be read from input video: {args.source}")

    elapsed = time.perf_counter() - started_at
    print(f"Saved {frames_processed} annotated frames to {output_path}")
    print(f"Average processing FPS: {frames_processed / elapsed:.2f}")


if __name__ == "__main__":
    main()
