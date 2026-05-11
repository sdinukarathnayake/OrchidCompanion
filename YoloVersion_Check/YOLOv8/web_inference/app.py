"""Browser and mobile inference: loads weights only. Never deletes trained_models/best.pt."""

import base64
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
YOLOV8_ROOT = APP_DIR.parent

DEFAULT_MODEL_CANDIDATES = [
    YOLOV8_ROOT / "trained_models" / "best.pt",
    YOLOV8_ROOT
    / "orchid_yolov8_runs"
    / "yolov8s_512_100epochs"
    / "weights"
    / "best.pt",
    YOLOV8_ROOT / "yolov8s.pt",
    APP_DIR / "best.pt",
    YOLOV8_ROOT / "best.pt",
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

    for candidate in YOLOV8_ROOT.rglob("best.pt"):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No weights found. Train first (trained_models/best.pt), place yolov8s.pt in the YOLOv8 folder, "
        "copy best.pt into web_inference/, or set MODEL_PATH to a .pt file."
    )


MODEL_PATH = _resolve_model_path()
MODEL = YOLO(str(MODEL_PATH))

app = Flask(__name__)


@app.context_processor
def _inject_model_meta():
    return {"model_filename": MODEL_PATH.name}


DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 512
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


def _aggregate_by_class(detections: list[dict]) -> list[dict]:
    """One row per class: highest confidence among boxes of that class, plus how many boxes."""
    grouped: dict[str, dict] = {}
    for det in detections:
        label = det["label"]
        conf = float(det["confidence"])
        group = grouped.setdefault(label, {"label": label, "max_conf": conf, "count": 0})
        group["count"] += 1
        group["max_conf"] = max(group["max_conf"], conf)

    rows = [
        {"label": label, "confidence": round(values["max_conf"], 4), "count": values["count"]}
        for label, values in grouped.items()
    ]
    rows.sort(key=lambda item: (-item["count"], -item["confidence"]))
    return rows


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


def _base_view_context(conf: float = DEFAULT_CONF) -> dict:
    return {
        "result_image": None,
        "detections": [],
        "detection_rows": [],
        "model_path": str(MODEL_PATH),
        "model_classes": _normalize_class_names(MODEL.names),
        "conf": conf,
        "imgsz": DEFAULT_IMGSZ,
        "diagnostics": None,
        "best_guess": None,
        "detection_hint": None,
        "error": None,
    }


def _render_index(prediction: dict | None = None, conf: float = DEFAULT_CONF, error: str | None = None):
    context = _base_view_context(conf=conf)
    if prediction is not None:
        context.update(prediction)
    context["error"] = error
    return render_template("index.html", **context)


def _predict_from_image(image, conf: float) -> dict:
    height, width = image.shape[:2]
    results = MODEL.predict(source=image, conf=conf, imgsz=DEFAULT_IMGSZ, verbose=False)
    active_result = results[0]
    plotted = active_result.plot()

    success, encoded = cv2.imencode(".jpg", plotted)
    if not success:
        raise RuntimeError("Failed to render model output image.")

    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    detections = _extract_detections(active_result)

    detection_hint = None
    best_guess = None
    diagnostics = {
        "image_size": f"{width}x{height}",
        "boxes_at_selected_conf": len(detections),
        "boxes_at_probe_conf": None,
        "max_conf_at_probe": None,
        "adaptive_conf_used": None,
        "best_guess_label": None,
        "best_guess_confidence": None,
        "best_guess_support": None,
    }

    if not detections:
        probe_result = MODEL.predict(source=image, conf=0.001, imgsz=DEFAULT_IMGSZ, verbose=False)[0]
        probe_detections = _extract_detections(probe_result)
        best_guess = _summarize_best_guess(probe_detections)
        probe_boxes = probe_result.boxes
        probe_count = 0
        probe_max_conf = 0.0

        if probe_boxes is not None and probe_boxes.conf is not None:
            probe_confidences = [float(value) for value in probe_boxes.conf.tolist()]
            probe_count = int(len(probe_confidences))
            if probe_confidences:
                probe_max_conf = max(probe_confidences)

        diagnostics["boxes_at_probe_conf"] = probe_count
        diagnostics["max_conf_at_probe"] = round(probe_max_conf, 4)
        diagnostics["best_guess_label"] = best_guess["label"] if best_guess else None
        diagnostics["best_guess_confidence"] = best_guess["confidence"] if best_guess else None
        diagnostics["best_guess_support"] = best_guess["count"] if best_guess else None

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

        if not detections and detection_hint is None:
            detection_hint = (
                "No boxes passed the threshold. If probe count is also 0, the model likely does not "
                "recognize this image/class. Try a training-like image or retraining with more similar data."
            )

    detection_rows = _aggregate_by_class(detections)

    return {
        "result_image": image_b64,
        "detections": detections,
        "detection_rows": detection_rows,
        "model_path": str(MODEL_PATH),
        "model_classes": _normalize_class_names(MODEL.names),
        "conf": conf,
        "imgsz": DEFAULT_IMGSZ,
        "diagnostics": diagnostics,
        "best_guess": best_guess,
        "detection_hint": detection_hint,
        "image_size": {"width": width, "height": height},
    }


def _predict_from_upload(upload, conf: float):
    if upload is None or upload.filename == "":
        return None, "Please choose an image file."

    raw_bytes = upload.read()
    if not raw_bytes:
        return None, "Uploaded file is empty."

    np_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None, "Unsupported image format. Use jpg, jpeg, png, or webp."

    try:
        return _predict_from_image(image, conf), None
    except RuntimeError as exc:
        return None, str(exc)


@app.get("/")
def index():
    return _render_index()


@app.post("/predict")
def predict():
    conf = _parse_confidence(request.form.get("conf"))
    prediction, error = _predict_from_upload(request.files.get("image"), conf)
    return _render_index(prediction=prediction, conf=conf, error=error)


@app.get("/api/health")
def api_health():
    return jsonify(
        {
            "ok": True,
            "modelFilename": MODEL_PATH.name,
            "modelPath": str(MODEL_PATH),
            "modelClasses": _normalize_class_names(MODEL.names),
            "defaultConf": DEFAULT_CONF,
            "imgsz": DEFAULT_IMGSZ,
        }
    )


@app.post("/api/predict")
def api_predict():
    conf = _parse_confidence(request.form.get("conf"))
    prediction, error = _predict_from_upload(request.files.get("image"), conf)
    if error:
        return jsonify({"ok": False, "error": error}), 400

    return jsonify(
        {
            "ok": True,
            "modelFilename": MODEL_PATH.name,
            "modelPath": prediction["model_path"],
            "modelClasses": prediction["model_classes"],
            "conf": prediction["conf"],
            "imgsz": prediction["imgsz"],
            "imageSize": prediction["image_size"],
            "resultImage": prediction["result_image"],
            "detections": prediction["detections"],
            "detectionRows": prediction["detection_rows"],
            "diagnostics": prediction["diagnostics"],
            "bestGuess": prediction["best_guess"],
            "detectionHint": prediction["detection_hint"],
        }
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5008"))
    app.run(host=host, port=port, debug=False)
