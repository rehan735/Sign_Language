
import os, sys
import numpy as np
import cv2
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ─── silence TF ──────────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ─── labels (35 classes) ─────────────────────────────────────────────────────
# Sorted: digits 1-9 then letters A-Z  (matches dataset/Indian/ folder names)
LABELS = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z",
]

# MediaPipe hand connection pairs (same as HAND_CONNECTIONS in JS)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm
]

IMG_SIZE = 256   # model input dimension

# ─── model loading ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATHS = [
    BASE_DIR / "Model" / "best_model.keras",
    BASE_DIR / "Model" / "Prediction_Model.h5",
]

print("Loading TensorFlow / Keras model …", flush=True)
try:
    import tensorflow as tf  # noqa: E402

    model = None
    for path in MODEL_PATHS:
        if path.exists():
            print(f"  Found model: {path.name}", flush=True)
            model = tf.keras.models.load_model(str(path), compile=False)
            print(f"  Input  shape: {model.input_shape}", flush=True)
            print(f"  Output shape: {model.output_shape}", flush=True)
            break

    if model is None:
        print("ERROR: no model file found in backend/Model/", file=sys.stderr)
        sys.exit(1)

except Exception as exc:
    print(f"ERROR loading model: {exc}", file=sys.stderr)
    sys.exit(1)


# ─── helpers ─────────────────────────────────────────────────────────────────

def landmarks_to_image(lm_flat: np.ndarray) -> np.ndarray:

    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    for hand_idx in range(2):
        offset = hand_idx * 63
        hand = lm_flat[offset: offset + 63].reshape(21, 3)

        # Skip if wrist is all-zero (hand absent)
        if np.all(hand[0] == 0):
            continue

        pts = []
        for lm in hand:
            x = int(np.clip(lm[0], 0, 1) * (IMG_SIZE - 1))
            y = int(np.clip(lm[1], 0, 1) * (IMG_SIZE - 1))
            pts.append((x, y))

        # Draw connections
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], 200, 2)

        # Draw landmark dots
        for p in pts:
            cv2.circle(canvas, p, 3, 255, -1)

    return canvas[..., np.newaxis]   # (256, 256, 1)


def best_frame_image(frames: list[list[float]]) -> np.ndarray | None:

    arr = np.array(frames, dtype=np.float32)   # (N, 126)

    # Score each frame by total landmark activity
    scores = np.abs(arr).sum(axis=1)
    best_idx = int(np.argmax(scores))
    best = arr[best_idx]

    # Skip totally blank frames
    if scores[best_idx] == 0:
        return None

    img = landmarks_to_image(best)
    img_norm = img.astype(np.float32) / 255.0
    return img_norm[np.newaxis, ...]   # (1, 256, 256, 1)


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="ISL Prediction Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    frames: list[list[float]]


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames array is empty")

    try:
        batch = best_frame_image(req.frames)
        if batch is None:
            raise HTTPException(status_code=400, detail="No hand landmarks detected in any frame")

        preds = model.predict(batch, verbose=0)      # (1, 35)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx])
        label = LABELS[class_idx] if class_idx < len(LABELS) else str(class_idx)

        return PredictResponse(prediction=label, confidence=confidence)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
