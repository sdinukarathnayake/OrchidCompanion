"""Local YOLO26 training (50 epochs by default). Mirrors YOLO26_Colab_Training.ipynb.

- Unzips YOLO26.zip (or first *.zip) in this folder into ./yolo26_dataset
- Picks data.yaml like the Colab notebook (prefers yaml next to train/ + valid or val)
- Trains with yolo26s.pt (override with YOLO_PRETRAINED)
- Saves runs under ./yolo26_runs/<run_name>/
- Copies weights/best.pt -> ./trained_models/best.pt (optional backup of previous best)

Env: YOLO_EPOCHS (default 50), YOLO_BATCH, YOLO_RUN_NAME, YOLO_PRETRAINED, YOLO_NO_BACKUP=1
"""
from __future__ import annotations

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


def _backup_existing_best(dest: Path) -> Path | None:
    if not dest.is_file():
        return None
    folder = dest.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = folder / f"best_before_{stamp}.pt"
    n = 1
    while backup.exists():
        backup = folder / f"best_before_{stamp}_{n}.pt"
        n += 1
    shutil.copy2(dest, backup)
    return backup


def _yaml_score(p: Path) -> tuple:
    d = p.parent
    has_train = (d / "train" / "images").is_dir()
    has_val_split = (d / "valid" / "images").is_dir() or (d / "val" / "images").is_dir()
    return (0 if has_train and has_val_split else 1, len(p.parts), str(p))


def _pick_data_yaml(extract_dir: Path) -> Path:
    candidates = list(extract_dir.rglob("data.yaml"))
    if not candidates:
        raise SystemExit("data.yaml not found after unzip.")
    chosen = sorted(candidates, key=_yaml_score)[0]
    if len(candidates) > 1:
        print("Multiple data.yaml found; using:", chosen)
    return chosen


def _validate_dataset_layout(dataset_dir: Path) -> None:
    _val_root = dataset_dir / "valid"
    if not (_val_root / "images").is_dir():
        _alt = dataset_dir / "val"
        if (_alt / "images").is_dir():
            _val_root = _alt
            print("Using val/ instead of valid/ for validation split.")
        else:
            _val_root = dataset_dir / "valid"

    required = [
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "labels",
        _val_root / "images",
        _val_root / "labels",
        dataset_dir / "test" / "images",
        dataset_dir / "test" / "labels",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing dataset folders:\n" + "\n".join(missing))


def main() -> None:
    base = Path(__file__).resolve().parent
    extract_dir = base / "yolo26_dataset"
    run_project = base / "yolo26_runs"

    zip_candidates = [base / "YOLO26.zip", *sorted(base.glob("*.zip"))]
    zip_path = next((p for p in zip_candidates if p.exists()), None)
    if zip_path is None:
        raise SystemExit(f"No dataset .zip found in {base} (expected YOLO26.zip).")

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    data_yaml_path = _pick_data_yaml(extract_dir)
    dataset_dir = data_yaml_path.parent
    data_yaml = str(data_yaml_path)
    _validate_dataset_layout(dataset_dir)

    device = 0 if torch.cuda.is_available() else "cpu"
    epochs = int(os.environ.get("YOLO_EPOCHS", "50"))
    batch = int(os.environ.get("YOLO_BATCH", "8"))
    if device == "cpu":
        batch = min(batch, 4)

    pretrained = os.environ.get("YOLO_PRETRAINED", "yolo26s.pt").strip() or "yolo26s.pt"
    run_name = os.environ.get("YOLO_RUN_NAME", f"yolo26s_512_{epochs}epochs").strip() or f"yolo26s_512_{epochs}epochs"

    print(f"Device: {device}, epochs: {epochs}, batch: {batch}")
    print(f"Pretrained: {pretrained}")
    print(f"data: {data_yaml}")
    print(f"Saving runs to: {run_project / run_name}")

    model = YOLO(pretrained)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=512,
        batch=batch,
        patience=30,
        device=device,
        project=str(run_project),
        name=run_name,
        exist_ok=True,
    )

    best_src = run_project / run_name / "weights" / "best.pt"
    if not best_src.is_file():
        raise SystemExit(f"Missing weights: {best_src}")

    dest = base / "trained_models" / "best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("YOLO_NO_BACKUP", "").strip() not in ("1", "true", "yes"):
        prev = _backup_existing_best(dest)
        if prev is not None:
            print(f"Backed up previous weights to: {prev.resolve()}")
    shutil.copy2(best_src, dest)
    print(f"Updated trained_models/best.pt (copy from run): {dest.resolve()}")


if __name__ == "__main__":
    main()
