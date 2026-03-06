# Chess Atlas

Chess piece recognition from YouTube chess content. Two-stage CV pipeline:

1. **Board detector** — YOLOv8n locates the chessboard bounding box in a video frame
2. **Piece classifier** — MobileNetV3-Small classifies each of the 64 squares into 13 classes

| Metric | Value |
|---|---|
| Test macro F1 | **0.9994** |
| Test weighted F1 | 0.9997 |
| Training samples | 112,512 |
| Classes | 13 (12 piece types + Empty) |
| Classifier size | 4.8 MB (TorchScript) |
| Board detector | 6.0 MB (ONNX, 320px) |

---

## Architecture

```
Video frame (640×360)
       │
       ▼
 YOLOv8n board detector          ← ONNX Runtime, 320px input, ~5ms CPU
       │  bbox (x1,y1,x2,y2)
       ▼
 Axis-aligned crop + resize      ← 512×512 square
       │
       ▼
 8×8 grid slice                  ← 64 crops of 64×64px
       │
       ▼
 MobileNetV3-Small classifier    ← batch of 64, TorchScript, 224×224 input
       │  13-class softmax × 64
       ▼
 Board state (FEN-compatible)
```

**Classes:** BlackBishop, BlackKing, BlackKnight, BlackPawn, BlackQueen, BlackRook, Empty, WhiteBishop, WhiteKing, WhiteKnight, WhitePawn, WhiteQueen, WhiteRook

---

## Project structure

```
ChessCVModel/
├── data_collection/           # Data pipeline (download → label → review → export)
│   ├── cli.py                 # Unified CLI entry point
│   ├── config.py              # All paths and thresholds (Config dataclass)
│   ├── download_videos.py     # yt-dlp YouTube downloader
│   ├── extract_frames.py      # Frame extraction at N-second intervals
│   ├── detect_boards.py       # YOLOv8 board detection on frames
│   ├── crop_boards.py         # Bbox crop + resize to square
│   ├── slice_squares.py       # 8×8 grid slicer → 64 square crops
│   ├── pseudo_label.py        # Classifier inference → auto-labels
│   ├── build_review_queue.py  # Build review_queue.csv for human review
│   ├── review_server.py       # Flask HTML review UI (keyboard-driven)
│   ├── export_dataset.py      # Export verified labels → split_assignments.csv
│   └── qa_report.py           # Dataset quality checks
│
├── board_detection/           # YOLO training pipeline
│   ├── 01_download_videos.py
│   ├── 02_extract_frames.py
│   ├── 03_auto_label.py       # Template-matching auto-labeler for YOLO
│   ├── 04_review_labels.py
│   ├── 05_prepare_dataset.py
│   ├── 06_train_yolo.ipynb    # YOLOv8n fine-tuning
│   └── runs/detect/runs/yolov8n_chessboard/weights/
│       ├── best.pt            # PyTorch weights
│       └── best_320.onnx      # ONNX export at 320px (use in production)
│
├── chess_training.ipynb       # MobileNetV3-Small training pipeline
├── chess_split.ipynb          # Train/val/test split assignment
├── ExploratoryDataAnalysis-EDA.ipynb
│
├── runs/                      # Training run outputs (timestamped)
│   └── <run_id>/
│       ├── chess_atlas_v1.torchscript.pt   # Inference model
│       ├── chess_atlas_v1.onnx             # ONNX export
│       ├── run_summary.json                # Norm stats, label map, metrics
│       ├── norm_stats.json
│       ├── history.json
│       └── config.json
│
├── data/final_dataset/
│   ├── split_assignments.csv  # filepath, label, split (train/val/test)
│   └── images/                # Square crop PNGs
│
├── test_samples/              # Input images for test_inference.py
├── test_results/              # Annotated output images
└── test_inference.py          # End-to-end inference script
```

---

## Setup

```bash
conda create -n chesscv python=3.12
conda activate chesscv
pip install torch torchvision ultralytics onnxruntime opencv-python \
            flask pandas scikit-learn matplotlib seaborn yt-dlp
```

Python 3.12, PyTorch 2.10, ONNX Runtime 1.24.

---

## Data pipeline

The full pipeline is driven by a single CLI. All stages are idempotent — re-running skips already-processed items.

```bash
# Run all stages for a list of YouTube video IDs
python -m data_collection.cli run-all --ids "VIDEO_ID_1 VIDEO_ID_2"

# Or run stages individually
python -m data_collection.cli download       --ids "..."
python -m data_collection.cli extract-frames --ids "..." [--interval 30]
python -m data_collection.cli detect-boards  --ids "..."
python -m data_collection.cli crop-boards    --ids "..."
python -m data_collection.cli slice-squares  --ids "..."
python -m data_collection.cli pseudo-label   --ids "..."
python -m data_collection.cli build-review
python -m data_collection.cli qa
```

### Human review

After pseudo-labeling, launch the keyboard-driven review UI:

```bash
python -m data_collection.cli review --reviewer yourname
python -m data_collection.cli review --filter corrected   # re-check corrections
python -m data_collection.cli review --filter all --port 7861
```

The review server runs at `http://localhost:7860`. Keyboard shortcuts:

| Key | Action |
|---|---|
| `←` `→` `↑` `↓` | Move cursor |
| `Space` | Toggle selection + advance |
| `Enter` | Approve cursor square |
| `A` | Approve page (non-selected) |
| `Q` | Approve entire class |
| `C` | Correct selected squares (label picker) |
| `U` | Mark selected uncertain |
| `Esc` | Clear selection |
| `PgUp` / `PgDn` | Previous / next page |

### Export dataset

```bash
python -m data_collection.cli export --train-ratio 0.70 --val-ratio 0.15
```

Outputs `data/final_dataset/split_assignments.csv` and copies images into `data/final_dataset/images/`.

---

## Training

### Board detector (YOLOv8n)

Open and run `board_detection/06_train_yolo.ipynb`. After training, export to ONNX at 320px for production:

```python
from ultralytics import YOLO
model = YOLO("board_detection/runs/detect/runs/yolov8n_chessboard/weights/best.pt")
model.export(format="onnx", imgsz=320, simplify=True)
```

### Piece classifier (MobileNetV3-Small)

Open and run `chess_training.ipynb`. Key config in Cell 0:

```python
CFG = dict(
    unfreeze_epoch      = 10,    # freeze backbone for first N epochs, then fine-tune all
    sampler_power       = 0.75,  # class-balanced sampling (0=none, 1=full inverse freq)
    sampler_multiplier  = 1.0,   # epoch size multiplier
    aug_arrow_prob      = 0.40,  # streamer arrow/highlight overlay (up to fully opaque)
    aug_jpeg_prob       = 0.50,  # JPEG compression blockiness simulation
    ...
)
```

**Training augmentations:**
- JPEG compression artifacts (quality 15–85) — simulates video frame blockiness
- Streamer arrow/highlight overlays (α 0.3–1.0) — semi-transparent to fully opaque
- `RandomAffine` ±8% translation + ±8% scale — handles off-center board slicing
- `ColorJitter` brightness/contrast ±40% + `RandomAutocontrast` — varying stream conditions

Outputs a timestamped run under `runs/` containing `chess_atlas_v1.torchscript.pt` and `run_summary.json`.

---

## Inference

```bash
# Test on images in test_samples/  →  annotated PNGs in test_results/
python test_inference.py

python test_inference.py \
  --samples  test_samples/ \
  --run      runs/20260305_222242 \
  --yolo     board_detection/runs/detect/runs/yolov8n_chessboard/weights/best_320.onnx \
  --out      test_results/
```

### Using in a video pipeline

```python
from test_inference import FastBoardDetector, BBoxCache, load_classifier, \
                           build_transform, classify_squares, crop_board, slice_board
import json, torch
from pathlib import Path

summary    = json.load(open("runs/20260305_222242/run_summary.json"))
labels     = [k for k, _ in sorted(summary["label_map"].items(), key=lambda x: x[1])]
device     = torch.device("cpu")
classifier = load_classifier(Path("runs/20260305_222242"), device)
transform  = build_transform(summary["norm_mean"], summary["norm_std"], 224)
detector   = FastBoardDetector(Path("board_detection/.../best_320.onnx"))
cache      = BBoxCache(iou_threshold=0.85, max_misses=30)

# Per frame:
bbox        = cache.get(detector, frame_bgr)   # ~free when board hasn't moved
board_bgr   = crop_board(frame_bgr, bbox)
squares     = slice_board(board_bgr)
predictions = classify_squares(classifier, squares, transform, labels, device)
```

### CPU performance (Render Standard, 0.5 vCPU)

| Step | Approach | Est. latency |
|---|---|---|
| Board detection | ultralytics Python stack, 640px | ~500ms |
| Board detection | ONNX Runtime, 320px | ~100ms |
| Board detection | ONNX Runtime + BBoxCache hit | ~0ms |
| Piece classification | TorchScript, batch 64 | ~50ms |

`best_320.onnx` requires only `onnxruntime-cpu` — no PyTorch needed on the inference server.

---

## Results

Best run: `runs/20260305_222242`

```
Test macro F1    : 0.9994      Best epoch: 19 / 34
Test weighted F1 : 0.9997

Per-class test F1:
  BlackBishop  0.9990    BlackKing    1.0000    BlackKnight  0.9989
  BlackPawn    0.9991    BlackQueen   1.0000    BlackRook    0.9993
  Empty        0.9998    WhiteBishop  1.0000    WhiteKing    0.9987
  WhiteKnight  1.0000    WhitePawn    0.9996    WhiteQueen   0.9984
  WhiteRook    1.0000
```

Training split: 112,512 images (70% train / 15% val / 15% test), stratified by video source to prevent leakage across splits.
