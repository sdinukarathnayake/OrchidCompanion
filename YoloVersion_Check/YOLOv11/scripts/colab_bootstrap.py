#!/usr/bin/env python3
"""
Prepare a Colab-safe Ultralytics data config for this dataset.

This keeps local files untouched and writes a new YAML file that points to
absolute split directories under a chosen dataset root.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def parse_names_and_nc(source_yaml: Path) -> tuple[int, list[str]]:
    nc = None
    names = None

    for raw_line in source_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("nc:"):
            nc = int(line.split(":", 1)[1].strip())
        elif line.startswith("names:"):
            names = ast.literal_eval(line.split(":", 1)[1].strip())

    if nc is None or names is None:
        raise ValueError(f"Could not parse 'nc' and 'names' from {source_yaml}")

    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError("Expected 'names' to be a list of class names")

    if nc != len(names):
        raise ValueError(f"'nc' ({nc}) does not match number of names ({len(names)})")

    return nc, names


def ensure_split_structure(dataset_root: Path) -> None:
    required = [
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        dataset_root / "valid" / "images",
        dataset_root / "valid" / "labels",
        dataset_root / "test" / "images",
        dataset_root / "test" / "labels",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_str = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "Dataset structure is incomplete. Missing:\n"
            f"{missing_str}\n\n"
            "Expected folders under dataset root:\n"
            "train/images, train/labels, valid/images, valid/labels, test/images, test/labels"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Colab-safe data YAML for Ultralytics YOLO11 training."
    )
    parser.add_argument(
        "--dataset-root",
        default="/content/YOLOv11",
        help="Dataset root that contains train/ valid/ test folders.",
    )
    parser.add_argument(
        "--source-data",
        default="data.yaml",
        help="Path to source data.yaml file to read class metadata from.",
    )
    parser.add_argument(
        "--output",
        default="data.colab.yaml",
        help="Output YAML path.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    source_data = Path(args.source_data).resolve()
    output = Path(args.output).resolve()

    if not source_data.exists():
        raise FileNotFoundError(f"Source YAML not found: {source_data}")

    ensure_split_structure(dataset_root)
    nc, names = parse_names_and_nc(source_data)

    lines = [
        f"train: {dataset_root / 'train' / 'images'}",
        f"val: {dataset_root / 'valid' / 'images'}",
        f"test: {dataset_root / 'test' / 'images'}",
        "",
        f"nc: {nc}",
        f"names: {names}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] Wrote: {output}")
    print("[next] Example train command:")
    print(
        "yolo task=detect mode=train "
        "model=yolo11n.pt "
        f"data={output} epochs=100 imgsz=640 batch=16 device=0"
    )


if __name__ == "__main__":
    main()
