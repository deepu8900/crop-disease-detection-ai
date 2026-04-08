from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os, sys, threading, time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv_app.live_detection as _ld
from cv_app.live_detection import (
    has_leaf_colour,
    predict_frame,
    predict_and_annotate,
    get_best_result,
    reset_best_result,
    class_names as CLASS_NAMES,
)

app = FastAPI(title="Crop Disease Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
def read_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/health")
def health_check():
    return {"status": "ok", "classes": len(CLASS_NAMES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """File upload — green check + disease model."""
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Upload JPEG, PNG, or WebP only.")
    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        if not has_leaf_colour(frame):
            return {"disease": "No leaf detected", "confidence": 0.0, "status": "no_leaf"}
        label, confidence = predict_frame(frame)
        if label == "background":
            return {"disease": "No leaf detected", "confidence": 0.0, "status": "no_leaf"}
        return {
            "disease":    label,
            "confidence": round(confidence * 100, 2),
            "status":     "ok",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@app.get("/latest_result")
def latest_result():
    result = get_best_result()
    return {
        "disease":      result["disease"],
        "confidence":   result["confidence"],
        "has_snapshot": result["snapshot"] is not None,
    }


@app.get("/snapshot")
def snapshot():
    result = get_best_result()
    if result["snapshot"] is None:
        raise HTTPException(status_code=404, detail="No snapshot available yet.")
    return StreamingResponse(iter([result["snapshot"]]), media_type="image/jpeg")


@app.post("/reset_result")
def reset_result():
    reset_best_result()
    return {"status": "reset"}

_frame_lock  = threading.Lock()
_result_lock = threading.Lock()

_latest_frame  = {"frame": None, "ready": False}
_latest_label  = {"text": "Scanning for leaf...", "color": (200, 200, 0)}


def _prediction_worker():

    while True:
        
        frame = None
        with _frame_lock:
            if _latest_frame["ready"]:
                frame = _latest_frame["frame"].copy()
                _latest_frame["ready"] = False

        if frame is None:
            time.sleep(0.01)   
            continue

        text, color = predict_and_annotate(frame)

        with _result_lock:
            _latest_label["text"]  = text
            _latest_label["color"] = color

_pred_thread = threading.Thread(target=_prediction_worker, daemon=True)
_pred_thread.start()


def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    SUBMIT_EVERY = 8   
    frame_count  = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            if frame_count % SUBMIT_EVERY == 0:
                with _frame_lock:
                    _latest_frame["frame"] = frame.copy()
                    _latest_frame["ready"] = True

            with _result_lock:
                text  = _latest_label["text"]
                color = _latest_label["color"]

            display = frame.copy()
            font    = cv2.FONT_HERSHEY_SIMPLEX
            scale, thick = 0.6, 2
            (tw, th), _  = cv2.getTextSize(text, font, scale, thick)
            cv2.rectangle(display, (15,15), (25+tw, 25+th), (0,0,0), -1)
            cv2.putText(display, text, (20, th+18), font, scale, color, thick)

            ret, buffer = cv2.imencode(
                ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 65])
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
