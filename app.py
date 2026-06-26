cat << 'EOF' > app.py
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))

app = FastAPI(title="SMT Detection SD Inference API")

model: YOLO | None = None

@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_path": str(MODEL_PATH),
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    suffix = Path(file.filename or "").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await file.read())

    try:
        results = model.predict(
            source=str(temp_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            save=False,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": float(boxes.conf[index]),
                    "bbox": [float(value) for value in boxes.xyxy[index].tolist()],
                }
            )

    return {
        "filename": file.filename,
        "model_path": str(MODEL_PATH),
        "detections": detections,
        "count": len(detections),
    }
EOF