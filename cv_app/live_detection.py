import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
import threading

ROOT_DIR        = Path(__file__).resolve().parent.parent
MODEL_PATH      = ROOT_DIR / "models" / "crop_disease_mobilenet.h5"
NAMES_PATH      = ROOT_DIR / "cv_app"  / "class_names.txt"

YOLO_MODEL_NAME = str(ROOT_DIR / "models" / "yolov8n.pt")   
IMG_SIZE             = 224
CONFIDENCE_THRESHOLD = 0.75

GREEN_RANGES = [
    (np.array([22,  40,  40], dtype=np.uint8),
     np.array([95, 255, 255], dtype=np.uint8)),
    (np.array([15,  40,  40], dtype=np.uint8),
     np.array([22, 255, 255], dtype=np.uint8)),
]
LEAF_PIXEL_MIN    = 0.06

YOLO_CONF         = 0.07  
YOLO_PADDING      = 10     

YOLO_PLANT_CLASSES = {58, 60, 62} 

COLOR_OK          = (0, 220, 80)
COLOR_LOW         = (0, 180, 255)
COLOR_SCAN        = (200, 200, 0)
COLOR_BG          = (0, 0, 0)
COLOR_BBOX        = (0, 255, 120)

print(f"[*] Loading disease model from {MODEL_PATH} …")
model: tf.keras.Model = tf.keras.models.load_model(str(MODEL_PATH))
print("[✓] Disease model ready.")

yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_MODEL_NAME)
    print("[✓] YOLOv8n pretrained (COCO) loaded — no training needed.")
except ImportError:
    print("[!] ultralytics not installed — run: pip install ultralytics")
    print("    → Falling back to green colour check only.")
except Exception as e:
    print(f"[!] YOLO load failed: {e}")
    print("    → Falling back to green colour check only.")

if NAMES_PATH.exists():
    with open(NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"[✓] Loaded {len(class_names)} class names.")
else:
    print(f"[!] {NAMES_PATH} not found — using built-in class list.")
    class_names = [
        "Apple___Apple_scab", "Apple___Black_rot",
        "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
        "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
        "Potato___Early_blight", "Potato___Late_blight",
        "Potato___healthy", "Raspberry___healthy",
        "Soybean___healthy", "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight",
        "Tomato___Late_blight", "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
    ]

_lock = threading.Lock()

best_result: dict = {
    "disease":    None,
    "confidence": 0.0,
    "snapshot":   None,
}

def update_best_result(label: str, confidence: float, frame: np.ndarray) -> None:
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with _lock:
        best_result["disease"]    = label
        best_result["confidence"] = round(confidence * 100, 2)
        best_result["snapshot"]   = buffer.tobytes()

def get_best_result() -> dict:
    with _lock:
        return dict(best_result)

def reset_best_result() -> None:
    with _lock:
        best_result["disease"]    = None
        best_result["confidence"] = 0.0
        best_result["snapshot"]   = None

def _green_check(frame: np.ndarray) -> bool:
    
    hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in GREEN_RANGES:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))
    return np.count_nonzero(combined) / combined.size >= LEAF_PIXEL_MIN

def _yolo_detect(frame: np.ndarray):
 
    if yolo_model is None:
        return None

    results = yolo_model(
        frame,
        conf    = YOLO_CONF,
        classes = list(YOLO_PLANT_CLASSES),  
        verbose = False,
    )
    boxes = []
    h, w  = frame.shape[:2]

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, x1 - YOLO_PADDING)
            y1 = max(0, y1 - YOLO_PADDING)
            x2 = min(w, x2 + YOLO_PADDING)
            y2 = min(h, y2 + YOLO_PADDING)
            boxes.append((x1, y1, x2, y2))

    return boxes


def has_leaf(frame: np.ndarray):
    if not _green_check(frame):
        return False   

    if yolo_model is None:
        return True   

    boxes = _yolo_detect(frame)
    return boxes       
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    img      = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    img      = np.expand_dims(img.astype(np.float32), axis=0)
    return preprocess_input(img)


def predict_frame(frame: np.ndarray) -> tuple[str, float]:
    """Run MobileNetV2 disease classifier on a BGR frame or crop."""
    preds      = model.predict(preprocess_frame(frame), verbose=0)
    idx        = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return f"Uncertain ({class_names[idx]})", confidence

    return class_names[idx], confidence

def _draw_label(frame: np.ndarray, text: str, color: tuple) -> None:
    font             = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.65, 2
    (tw, th), _      = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = 15, 15
    cv2.rectangle(frame, (x1, y1), (x1 + tw + 10, y1 + th + 10), COLOR_BG, -1)
    cv2.putText(frame, text, (x1 + 5, y1 + th + 3), font, scale, color, thickness)


def process_frame(frame: np.ndarray) -> np.ndarray:

    leaf_result = has_leaf(frame)

    if leaf_result is False or leaf_result == []:
        _draw_label(frame, "Scanning for leaf...", COLOR_SCAN)
        return frame

    if leaf_result is True:
        label, confidence = predict_frame(frame)
        _annotate(frame, label, confidence, bbox=None)
        return frame

    boxes = leaf_result
   
    best_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    x1, y1, x2, y2 = best_box

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        _draw_label(frame, "Scanning for leaf...", COLOR_SCAN)
        return frame

    label, confidence = predict_frame(crop)

    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)
    _annotate(frame, label, confidence, bbox=(x1, y1))

    return frame


def _annotate(frame, label, confidence, bbox=None):
    
    pct   = confidence * 100
    text  = f"{label}  {pct:.1f}%"
    color = COLOR_OK if confidence >= CONFIDENCE_THRESHOLD else COLOR_LOW

    if bbox:
        x1, y1 = bbox
        font             = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.65, 2
        (tw, th), _      = cv2.getTextSize(text, font, scale, thickness)
        lx = max(0, x1)
        ly = max(th + 15, y1 - 5)
        cv2.rectangle(frame, (lx, ly - th - 8), (lx + tw + 10, ly + 5), COLOR_BG, -1)
        cv2.putText(frame, text, (lx + 5, ly), font, scale, color, thickness)
    else:
        _draw_label(frame, text, color)

    bar_w = int(frame.shape[1] * confidence)
    cv2.rectangle(frame, (0, frame.shape[0]-6), (bar_w, frame.shape[0]), color, -1)

    with _lock:
        current_best = best_result["confidence"]
    if pct > current_best and confidence >= CONFIDENCE_THRESHOLD:
        update_best_result(label, confidence, frame.copy())

def run_live_detection(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[✗] Cannot open camera index {camera_index}.")
        return

    print("[✓] Webcam open. Press Q or ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            annotated = process_frame(frame)
            cv2.imshow("Crop Disease Detection  |  Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[✓] Camera released.")


if __name__ == "__main__":
    run_live_detection()
