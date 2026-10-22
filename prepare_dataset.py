"""Convert VisDrone-DET annotations into Ultralytics YOLO label files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
FIRST_OBJECT_CATEGORY = 1
LAST_OBJECT_CATEGORY = 10


def convert_annotation_line(line: str, image_width: int, image_height: int) -> str | None:
    """Return one normalized YOLO label, or None for an ignored/invalid box."""
    values = [value.strip() for value in line.split(",")]
    if len(values) < 6:
        raise ValueError(f"Expected at least 6 comma-separated values, got {len(values)}")

    x, y, width, height = (float(value) for value in values[:4])
    category = int(values[5])
    if category < FIRST_OBJECT_CATEGORY or category > LAST_OBJECT_CATEGORY:
        return None

    x1 = min(max(x, 0.0), float(image_width))
    y1 = min(max(y, 0.0), float(image_height))
    x2 = min(max(x + width, 0.0), float(image_width))
    y2 = min(max(y + height, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None

    center_x = (x1 + x2) / (2 * image_width)
    center_y = (y1 + y2) / (2 * image_height)
    box_width = (x2 - x1) / image_width
    box_height = (y2 - y1) / image_height
    return (
        f"{category - FIRST_OBJECT_CATEGORY} "
        f"{center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
    )


def find_image(images_dir: Path, stem: str) -> Path:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        image_path = images_dir / f"{stem}{suffix}"
        if image_path.is_file():
            return image_path
    raise FileNotFoundError(f"No supported image found for annotation {stem!r} in {images_dir}")


def convert_split(split_dir: Path) -> tuple[int, int]:
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotations_dir}")

    annotation_files = sorted(annotations_dir.glob("*.txt"))
    if not annotation_files:
        raise FileNotFoundError(f"No annotation .txt files found in {annotations_dir}")

    labels_dir = split_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    box_count = 0

    for annotation_path in annotation_files:
        image_path = find_image(images_dir, annotation_path.stem)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image_height, image_width = image.shape[:2]

        labels = []
        with annotation_path.open("r", encoding="utf-8-sig") as annotation_file:
            for line_number, line in enumerate(annotation_file, start=1):
                if not line.strip():
                    continue
                try:
                    label = convert_annotation_line(line, image_width, image_height)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid annotation at {annotation_path}:{line_number}: {error}"
                    ) from error
                if label is not None:
                    labels.append(label)

        (labels_dir / f"{annotation_path.stem}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""),
            encoding="utf-8",
        )
        image_count += 1
        box_count += len(labels)

    return image_count, box_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/VisDrone"),
        help="Root containing train/ and val/ split directories (default: datasets/VisDrone)",
    )
    args = parser.parse_args()

    for split in ("train", "val"):
        images, boxes = convert_split(args.dataset_root / split)
        print(f"{split}: converted {images} images and {boxes} boxes")


if __name__ == "__main__":
    main()
