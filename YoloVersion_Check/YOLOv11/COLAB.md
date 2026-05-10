# Google Colab Training (YOLO11)

This project is a Roboflow YOLO-format dataset with split folders:
- `train/images`, `train/labels`
- `valid/images`, `valid/labels`
- `test/images`, `test/labels`

The repo `data.yaml` uses parent-relative paths (`../train/images`), which may fail in Colab depending on where you run commands.  
Use the bootstrap script below to generate `data.colab.yaml` with absolute Linux paths.

## 1) Start Colab runtime

- Runtime -> Change runtime type -> **GPU**
- Verify:

```bash
!nvidia-smi
```

## 2) Get project files into Colab

### Option A: Upload zip

```bash
%cd /content
!unzip -o YOLOv11.zip -d /content/YOLOv11
%cd /content/YOLOv11
```

### Option B: Use Google Drive (recommended for persistence)

```bash
from google.colab import drive
drive.mount('/content/drive')
```

If your dataset is at `/content/drive/MyDrive/YOLOv11`, run:

```bash
%cd /content/drive/MyDrive/YOLOv11
```

## 3) Install dependencies

```bash
!python -m pip install --upgrade pip
!pip install -U ultralytics
```

## 4) Generate Colab-safe data config

If you are in `/content/YOLOv11`:

```bash
!python scripts/colab_bootstrap.py \
  --dataset-root /content/YOLOv11 \
  --source-data data.yaml \
  --output data.colab.yaml
```

If you are in Drive (example path shown):

```bash
!python scripts/colab_bootstrap.py \
  --dataset-root /content/drive/MyDrive/YOLOv11 \
  --source-data data.yaml \
  --output data.colab.yaml
```

## 5) Train YOLO11 detection model

```bash
!yolo task=detect mode=train \
  model=yolo11n.pt \
  data=data.colab.yaml \
  epochs=100 imgsz=640 batch=16 device=0 \
  project=runs_colab name=orchid_yolo11n
```

## 6) Validate, infer, and export

Validate:

```bash
!yolo task=detect mode=val \
  model=runs_colab/orchid_yolo11n/weights/best.pt \
  data=data.colab.yaml device=0
```

Predict (replace image path as needed):

```bash
!yolo task=detect mode=predict \
  model=runs_colab/orchid_yolo11n/weights/best.pt \
  source=test/images device=0
```

Export:

```bash
!yolo mode=export \
  model=runs_colab/orchid_yolo11n/weights/best.pt \
  format=onnx
```

## 7) Save trained weights

If using Drive, outputs are already persistent when training inside Drive path.  
If training in `/content`, copy artifacts to Drive:

```bash
!cp -r runs_colab /content/drive/MyDrive/YOLOv11_runs_backup
```

## Cursor Colab extension note

If you use a Cursor Colab integration/extension, use it to:
- open the notebook/runtime,
- run the same shell commands above in order,
- and ensure runtime is GPU before training.

If extension features differ, the command sequence in this file is the fallback and is sufficient.
