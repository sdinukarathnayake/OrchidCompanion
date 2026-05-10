import base64
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = [
    APP_DIR / "best.pt",
    APP_DIR.parent / "best.pt",
]


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists() and path.is_file():
            return path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists() and candidate.is_file():
            return candidate

    # Last fallback: search nearby for a trained checkpoint.
    for candidate in APP_DIR.parent.rglob("best.pt"):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No best.pt model found. Place best.pt in YOLOv11/web_inference or set MODEL_PATH."
    )


MODEL_PATH = _resolve_model_path()
MODEL = YOLO(str(MODEL_PATH))

app = Flask(__name__)

DEFAULT_CONF = 0.10
DEFAULT_IMGSZ = 960
MIN_RELIABLE_CONF = 0.05


def _parse_confidence(raw_value: str | None) -> float:
    try:
        value = float(raw_value) if raw_value is not None else DEFAULT_CONF
    except ValueError:
        value = DEFAULT_CONF
    return max(0.01, min(0.95, value))


def _normalize_class_names(names) -> list[str]:
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    if isinstance(names, list):
        return names
    return []


def _extract_detections(result) -> list[dict]:
    detections = []
    boxes = result.boxes
    if boxes is None:
        return detections

    names = result.names
    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        cls_name = names.get(int(cls_id), str(int(cls_id)))
        detections.append({"label": cls_name, "confidence": round(float(conf), 4)})
    return detections


def _summarize_best_guess(detections: list[dict]) -> dict | None:
    if not detections:
        return None

    grouped = {}
    counts = Counter()
    for det in detections:
        label = det["label"]
        conf = float(det["confidence"])
        counts[label] += 1
        summary = grouped.setdefault(label, {"label": label, "count": 0, "max_conf": 0.0})
        summary["count"] += 1
        summary["max_conf"] = max(summary["max_conf"], conf)

    best = max(grouped.values(), key=lambda item: (item["count"], item["max_conf"]))
    best["confidence"] = round(best["max_conf"], 4)
    return best


@app.get("/")
def index():
    return render_template(
        "index.html",
        result_image=None,
        detections=[],
        model_path=str(MODEL_PATH),
        model_classes=_normalize_class_names(MODEL.names),
        conf=DEFAULT_CONF,
        imgsz=DEFAULT_IMGSZ,
        diagnostics=None,
        best_guess=None,
        detection_hint=None,
        error=None,
    )


@app.post("/predict")
def predict():
    conf = _parse_confidence(request.form.get("conf"))
    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return render_template(
            "index.html",
            result_image=None,
            detections=[],
            model_path=str(MODEL_PATH),
            model_classes=_normalize_class_names(MODEL.names),
            conf=conf,
            imgsz=DEFAULT_IMGSZ,
            diagnostics=None,
            best_guess=None,
            detection_hint=None,
            error="Please choose an image file.",
        )

    raw_bytes = upload.read()
    np_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return render_template(
            "index.html",
            result_image=None,
            detections=[],
            model_path=str(MODEL_PATH),
            model_classes=_normalize_class_names(MODEL.names),
            conf=conf,
            imgsz=DEFAULT_IMGSZ,
            diagnostics=None,
            best_guess=None,
            detection_hint=None,
            error="Unsupported image format. Use jpg, jpeg, or png.",
        )

    h, w = image.shape[:2]
    results = MODEL.predict(source=image, conf=conf, imgsz=DEFAULT_IMGSZ, verbose=False)
    active_result = results[0]
    plotted = active_result.plot()

    success, encoded = cv2.imencode(".jpg", plotted)
    if not success:
        return render_template(
            "index.html",
            result_image=None,
            detections=[],
            model_path=str(MODEL_PATH),
            model_classes=_normalize_class_names(MODEL.names),
            conf=conf,
            imgsz=DEFAULT_IMGSZ,
            diagnostics=None,
            best_guess=None,
            detection_hint=None,
            error="Failed to render model output image.",
        )

    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    detections = _extract_detections(active_result)

    detection_hint = None
    best_guess = None
    diagnostics = {
        "image_size": f"{w}x{h}",
        "boxes_at_selected_conf": len(detections),
        "boxes_at_probe_conf": None,
        "max_conf_at_probe": None,
        "adaptive_conf_used": None,
    }

    if not detections:
        probe_result = MODEL.predict(source=image, conf=0.001, imgsz=DEFAULT_IMGSZ, verbose=False)[0]
        probe_detections = _extract_detections(probe_result)
        best_guess = _summarize_best_guess(probe_detections)
        probe_boxes = probe_result.boxes
        probe_count = 0
        probe_max_conf = 0.0
        if probe_boxes is not None and probe_boxes.conf is not None:
            probe_confidences = [float(v) for v in probe_boxes.conf.tolist()]
            probe_count = int(len(probe_confidences))
            if probe_confidences:
                probe_max_conf = max(probe_confidences)
        diagnostics["boxes_at_probe_conf"] = probe_count
        diagnostics["max_conf_at_probe"] = round(probe_max_conf, 4)
        diagnostics["best_guess_label"] = best_guess["label"] if best_guess else None
        diagnostics["best_guess_confidence"] = best_guess["confidence"] if best_guess else None
        diagnostics["best_guess_support"] = best_guess["count"] if best_guess else None

        # If predictions exist but are below selected threshold, auto-adjust once.
        if probe_count > 0 and probe_max_conf > 0:
            if probe_max_conf >= MIN_RELIABLE_CONF:
                adaptive_conf = max(0.01, min(0.95, probe_max_conf * 0.9))
                adaptive_result = MODEL.predict(
                    source=image,
                    conf=adaptive_conf,
                    imgsz=DEFAULT_IMGSZ,
                    verbose=False,
                    max_det=25,
                )[0]
                adaptive_detections = _extract_detections(adaptive_result)
                if adaptive_detections:
                    detections = adaptive_detections
                    diagnostics["adaptive_conf_used"] = round(adaptive_conf, 4)
                    plotted = adaptive_result.plot()
                    success, encoded = cv2.imencode(".jpg", plotted)
                    if success:
                        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
                    detection_hint = (
                        f"Auto-adjusted confidence to {adaptive_conf:.4f} because selected conf={conf:.4f} "
                        "returned no boxes."
                    )
                    if best_guess:
                        detection_hint += (
                            f" Fallback best guess: {best_guess['label']} "
                            f"(confidence {best_guess['confidence']:.4f}, support {best_guess['count']})."
                        )
            else:
                detection_hint = (
                    f"Model confidence is too low (max={probe_max_conf:.4f}). "
                    "Predictions are unreliable for this image. Use training-like images, improve labels, "
                    "or retrain with more epochs/data."
                )
                if best_guess:
                    detection_hint += (
                        f" Fallback best guess: {best_guess['label']} "
                        f"(confidence {best_guess['confidence']:.4f}, support {best_guess['count']})."
                    )

        if not detections:
            if detection_hint is None:
                detection_hint = (
                    "No boxes passed the threshold. If probe count is also 0, the model likely does not "
                    "recognize this image/class. Try a training-like image or retraining with more similar data."
                )

    return render_template(
        "index.html",
        result_image=image_b64,
        detections=detections,
        model_path=str(MODEL_PATH),
        model_classes=_normalize_class_names(MODEL.names),
        conf=conf,
        imgsz=DEFAULT_IMGSZ,
        diagnostics=diagnostics,
        best_guess=best_guess,
        detection_hint=detection_hint,
        error=None,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
