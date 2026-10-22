# Aerial Object Detection with YOLOv8

Train and run a YOLOv8 object detector on the ten annotated object categories in the VisDrone-DET dataset. The repository includes scripts to convert the original VisDrone annotations, fine-tune and evaluate an Ultralytics model, and annotate aerial video with OpenCV.

> This repository contains code and configuration, not the VisDrone dataset or trained weights. Metrics and video FPS depend on the data, model, hardware, and inference settings; run the commands below to measure them rather than relying on unverified benchmark claims.

## Features

- Converts VisDrone-DET annotations into normalized YOLO labels; ignores the dataset's ignored-region and other-category labels.
- Fine-tunes an Ultralytics YOLOv8 checkpoint with a larger default image size (`960`) to help retain small aerial objects.
- Validates the best checkpoint and reports measured `mAP@0.5` and `mAP@0.5:0.95`.
- Runs inference on video, draws class-labelled boxes, saves an annotated MP4, and reports measured processing FPS.

## Setup

Python 3.9 or later is recommended. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the VisDrone-DET train and validation splits from the [official VisDrone dataset page](https://github.com/VisDrone/VisDrone-Dataset) and arrange their contents as follows:

```text
datasets/VisDrone/
├── train/
│   ├── images/       # VisDrone training images
│   └── annotations/  # Matching VisDrone annotation .txt files
└── val/
    ├── images/
    └── annotations/
```

Each annotation file must have the same filename stem as its image and use the VisDrone comma-separated format:
`x, y, width, height, score, category, truncation, occlusion`.

Convert annotations into YOLO labels beside each image directory:

```bash
python prepare_dataset.py --dataset-root datasets/VisDrone
```

The generated layout is `train/labels/` and `val/labels/`. The source annotation files are left unchanged.

## Train and evaluate

```bash
python train.py
```

The default run fine-tunes `yolov8s.pt` for 100 epochs at 960 pixels. Training automatically uses an available GPU; override the settings as needed:

```bash
python train.py --model yolov8s.pt --epochs 100 --imgsz 960 --batch 8 --device 0
```

Use `--device cpu` to train on CPU. Run artifacts are saved under `runs/detect/visdrone/`; after training, the script validates the best checkpoint and prints the measured metrics.

The dataset configuration is in [`configs/visdrone.yaml`](configs/visdrone.yaml). Update its `path` if the dataset is stored elsewhere.

## Detect objects in video

```bash
python predict_video.py --source path/to/aerial_video.mp4
```

By default, inference loads `runs/detect/visdrone/weights/best.pt` and writes `aerial_video_detections.mp4` beside the source. Select a checkpoint, output path, confidence threshold, or device explicitly:

```bash
python predict_video.py --source input.mp4 --weights runs/detect/visdrone/weights/best.pt --output output.mp4 --conf 0.25 --device 0
```

The output video shows labelled boxes and the measured end-to-end processing FPS. Actual throughput varies with GPU/CPU, input resolution, and codec; it is not a fixed project benchmark.

## VisDrone categories

Class IDs follow the VisDrone-DET annotation categories, shifted to zero-based IDs for YOLO:

| YOLO ID | VisDrone category |
|---:|---|
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | tricycle |
| 7 | awning-tricycle |
| 8 | bus |
| 9 | motor |

VisDrone category IDs 0 (ignored regions) and 11 (others) are excluded from training labels.

## Results

No model weights or validation results are bundled. Run training to generate a checkpoint and metrics on your local validation split. Record the resulting `mAP@0.5`, `mAP@0.5:0.95`, and video FPS together with the hardware and inference settings used.

## Dataset citation

```bibtex
@article{zhu2021detection,
  title={Detection and tracking meet drones challenge},
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={11},
  pages={7380--7399},
  year={2021},
  publisher={IEEE}
}
```
