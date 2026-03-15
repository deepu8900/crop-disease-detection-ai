from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os, sys
import cv2
import numpy as np

# ── import shared helpers from cv_app so preprocessing is NEVER duplicated ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv_app.live_detection import has_leaf, predict_frame, process_frame, class_names as CLASS_NAMES

# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────
app = FastAPI(title="Crop Disease Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path  = os.path.join(BASE_DIR, "frontend")

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.get("/")
def read_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/health")
def health_check():
    return {"status": "ok", "classes": len(CLASS_NAMES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Upload JPEG, PNG, or WebP only.")

    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        if not has_leaf(frame):
            return {"disease": "No leaf detected", "confidence": 0.0, "status": "no_leaf"}

        label, confidence = predict_frame(frame)
        return {
            "disease":    label,
            "confidence": round(confidence * 100, 2),
            "status":     "ok",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


# ─────────────────────────────────────────────
#  Live Webcam Stream
# ─────────────────────────────────────────────
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated = process_frame(frame)

            ret, buffer = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
