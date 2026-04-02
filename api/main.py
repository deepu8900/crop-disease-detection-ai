from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv_app.live_detection import (
    has_leaf_colour,    
    has_leaf_yolo,       
    predict_frame,
    process_frame,
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

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path  = os.path.join(BASE_DIR, "frontend")

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

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

        if not has_leaf_colour(frame): 
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
    return StreamingResponse(
        iter([result["snapshot"]]),
        media_type="image/jpeg",
    )


@app.post("/reset_result")
def reset_result():
    reset_best_result()
    return {"status": "reset"}


import threading as _threading

_pred_lock   = _threading.Lock()
_latest_label   = {"text": None, "color": (200, 200, 0)}  # shared state
_frame_for_pred = {"frame": None, "ready": False}          # input to thread


def _prediction_worker():

    while True:
        with _pred_lock:
            if not _frame_for_pred["ready"]:
                pass
            else:
                frame = _frame_for_pred["frame"].copy()
                _frame_for_pred["ready"] = False

                if has_leaf_yolo(frame) is not False:  
                    label, confidence = predict_frame(frame)
                    pct   = confidence * 100
                    text  = f"{label}  {pct:.1f}%"
                    color = (0, 220, 80) if confidence >= 0.75 else (0, 180, 255)

                    from cv_app.live_detection import best_result, update_best_result
                    with __import__("cv_app.live_detection",
                                    fromlist=["_lock"])._lock:
                        current_best = best_result["confidence"]
                    if pct > current_best and confidence >= 0.75:
                        update_best_result(label, confidence, frame)

                    _latest_label["text"]  = text
                    _latest_label["color"] = color
                else:
                    _latest_label["text"]  = "Scanning for leaf..."
                    _latest_label["color"] = (200, 200, 0)


def _draw_overlay(frame: np.ndarray) -> np.ndarray:
 
    text  = _latest_label["text"]
    color = _latest_label["color"]

    if text is None:
        text  = "Scanning for leaf..."
        color = (200, 200, 0)

    font             = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.65, 2
    (tw, th), _      = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (15, 15), (25 + tw, 25 + th), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, th + 18), font, scale, color, thickness)
    return frame

_pred_thread = _threading.Thread(target=_prediction_worker, daemon=True)
_pred_thread.start()


def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    SUBMIT_EVERY = 5 
    frame_count  = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            if frame_count % SUBMIT_EVERY == 0:
                with _pred_lock:
                    _frame_for_pred["frame"] = frame.copy()
                    _frame_for_pred["ready"] = True

            annotated = _draw_overlay(frame.copy())

            ret, buffer = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
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
