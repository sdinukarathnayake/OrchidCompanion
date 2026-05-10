# YOLOv8 Orchid Stage Detection on Google Colab

This dataset is already in YOLOv8 format with these classes:

- `Bud_formation`
- `Flowering`
- `Mature_Cane`
- `Seedling`
- `Vegetative`

## 1. Prepare the Dataset

Zip this whole `YOLOv8` folder on your laptop and upload it to Google Drive.

Recommended Drive path:

```text
MyDrive/YOLOv8.zip
```

In Colab, select:

```text
Runtime > Change runtime type > T4 GPU
```

## 2. Install YOLOv8

```python
!pip install -q ultralytics

from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

## 3. Mount Google Drive and Unzip Dataset

```python
from google.colab import drive
drive.mount('/content/drive')

!rm -rf /content/orchid_yolov8
!mkdir -p /content/orchid_yolov8
!unzip -q /content/drive/MyDrive/YOLOv8.zip -d /content/orchid_yolov8

!ls -R /content/orchid_yolov8 | head
```

If the zip contains one top-level `YOLOv8` folder, use this dataset path:

```python
DATASET_DIR = "/content/orchid_yolov8/YOLOv8"
DATA_YAML = f"{DATASET_DIR}/data.yaml"
```

If your zip extracted the `train`, `valid`, `test`, and `data.yaml` files directly into `/content/orchid_yolov8`, use this instead:

```python
# DATASET_DIR = "/content/orchid_yolov8"
# DATA_YAML = f"{DATASET_DIR}/data.yaml"
```

## 4. Check the Dataset

```python
from pathlib import Path
from collections import Counter

names = ['Bud_formation', 'Flowering', 'Mature_Cane', 'Seedling', 'Vegetative']
root = Path(DATASET_DIR)

for split in ["train", "valid", "test"]:
    image_count = len(list((root / split / "images").glob("*.jpg")))
    label_files = list((root / split / "labels").glob("*.txt"))
    counts = Counter()

    for label_file in label_files:
        for line in label_file.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1

    print(f"\n{split}: {image_count} images, {len(label_files)} labels")
    for class_id, class_name in enumerate(names):
        print(f"  {class_id}: {class_name}: {counts[class_id]}")
```

## 5. Train the Model

Start with `yolov8s.pt`. If training is still slow, use `yolov8n.pt`. If accuracy is poor and Colab time allows, try `yolov8m.pt`.

```python
model = YOLO("yolov8s.pt")

results = model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=512,
    batch=16,
    patience=30,
    device=0,
    project="/content/drive/MyDrive/orchid_yolov8_runs",
    name="yolov8s_512_100epochs",
    exist_ok=True
)
```

Your trained model will be saved in:

```text
MyDrive/orchid_yolov8_runs/yolov8s_512_100epochs/weights/best.pt
```

## 6. Validate and Test

```python
best_model = YOLO("/content/drive/MyDrive/orchid_yolov8_runs/yolov8s_512_100epochs/weights/best.pt")

best_model.val(data=DATA_YAML, split="val", imgsz=512, device=0)
best_model.val(data=DATA_YAML, split="test", imgsz=512, device=0)
```

## 7. Run Detection on Test Images

```python
best_model.predict(
    source=f"{DATASET_DIR}/test/images",
    imgsz=512,
    conf=0.25,
    save=True,
    project="/content/drive/MyDrive/orchid_yolov8_predictions",
    name="test_predictions",
    exist_ok=True
)
```

## Important Dataset Note

This folder currently has 420 training images, 3 validation images, and 12 test images.
The validation set is too small, and all validation labels are `Vegetative`.
For more trustworthy results, create a better validation set with examples from all 5 stages.
