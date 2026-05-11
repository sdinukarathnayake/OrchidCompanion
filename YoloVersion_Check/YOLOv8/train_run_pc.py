"""Local training runner — same layout as YOLOv8_Main.ipynb.

After training, copies the new run's weights/best.pt into trained_models/best.pt.
If trained_models/best.pt already exists, it is copied to trained_models/best_before_<timestamp>.pt
first (set YOLO_NO_BACKUP=1 to skip). The canonical path is then updated in place.
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
    """If dest exists, copy to best_before_<timestamp>.pt in the same folder. Returns new path or None."""
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


def main() -> None:
    base = Path(__file__).resolve().parent
    extract_dir = base / "orchid_yolov8"
    run_project = base / "orchid_yolov8_runs"
    run_name = "yolov8s_512_100epochs"

    zip_candidates = [
        base / "Orchid Stage Detection new.v2i.yolov8.zip",
        base / "YOLOv8.zip",
        *sorted(base.glob("*.zip")),
    ]
    zip_path = next((p for p in zip_candidates if p.exists()), None)
    if zip_path is None:
        raise SystemExit(f"No dataset .zip found in {base}")

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    yamls = list(extract_dir.rglob("data.yaml"))
    if not yamls:
        raise SystemExit("data.yaml not found after unzip")
    data_yaml = str(yamls[0])

    device = 0 if torch.cuda.is_available() else "cpu"
    epochs = int(os.environ.get("YOLO_EPOCHS", "50"))
    batch = int(os.environ.get("YOLO_BATCH", "8"))
    if device == "cpu":
        batch = min(batch, 4)

    print(f"Device: {device}, epochs: {epochs}, batch: {batch}")
    print(f"data: {data_yaml}")

    model = YOLO("yolov8s.pt")
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
