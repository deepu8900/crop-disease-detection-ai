import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path

# ─────────────────────────────────────────────
#  Paths  (resolved relative to project root,
#          so the file works from any cwd)
# ─────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent   # project root
MODEL_PATH = ROOT_DIR / "models" / "crop_disease_mobilenet.h5"
NAMES_PATH = ROOT_DIR / "cv_app"  / "class_names.txt"

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
IMG_SIZE             = 224
CONFIDENCE_THRESHOLD = 0.75
GREEN_THRESHOLD      = 0.05
GREEN_LOWER          = np.array([25,  40,  40],  dtype=np.uint8)
GREEN_UPPER          = np.array([85, 255, 255],  dtype=np.uint8)

# ── Overlay colours (BGR) ──────────────────────
COLOR_OK      = (0, 220, 80)    # green  — confident prediction
COLOR_LOW     = (0, 180, 255)   # orange — low confidence
COLOR_SCAN    = (200, 200, 0)   # cyan   — scanning / no leaf
COLOR_BG      = (0, 0, 0)       # black  — text background box

# ─────────────────────────────────────────────
#  Model + Class Names  (loaded once)
# ─────────────────────────────────────────────
print(f"[*] Loading model from {MODEL_PATH} …")
model: tf.keras.Model = tf.keras.models.load_model(str(MODEL_PATH))
print("[✓] Model ready.")

if NAMES_PATH.exists():
    with open(NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"[✓] Loaded {len(class_names)} class names.")
else:
    # ── Fallback: full PlantVillage 38-class list ──────────────
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

# ─────────────────────────────────────────────
#  Core helpers
# ─────────────────────────────────────────────
def has_leaf(frame: np.ndarray) -> bool:
    """
    Returns True when enough green pixels are present to suggest a leaf.
    Works on the raw frame (before brightness adjustment) so that leaf
    detection and prediction both see a consistent image.
    """
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    ratio = np.count_nonzero(mask) / mask.size   # BUG FIX: mask.size is faster
    return ratio > GREEN_THRESHOLD


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Shared preprocessing pipeline used by BOTH live_detection.py and
    main.py (via import) so predictions are always consistent.

    Pipeline: mild brightness/contrast boost → resize → MobileNetV2 norm.
    """
    # slight enhancement — keeps natural colours, helps low-light cams
    enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    img      = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    img      = np.expand_dims(img.astype(np.float32), axis=0)
    return preprocess_input(img)   # ← MobileNetV2 normalisation (−1 … 1)


def predict_frame(frame: np.ndarray) -> tuple[str, float]:
    """
    Run inference on a BGR frame.
    Returns (label, confidence_0_to_1).
    """
    preds      = model.predict(preprocess_frame(frame), verbose=0)
    idx        = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return f"Uncertain ({class_names[idx]})", confidence

    return class_names[idx], confidence


# ─────────────────────────────────────────────
#  Frame annotation
# ─────────────────────────────────────────────
def _draw_label(frame: np.ndarray, text: str, color: tuple) -> None:
    """Draw a filled-background label at the top-left of the frame."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.65, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = 15, 15
    # filled rectangle behind text so it's readable on any background
    cv2.rectangle(frame, (x1, y1), (x1 + tw + 10, y1 + th + 10), COLOR_BG, -1)
    cv2.putText(frame, text, (x1 + 5, y1 + th + 3), font, scale, color, thickness)


def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Annotate a single frame with the prediction overlay.
    Safe to call in a loop — returns the modified frame.
    """
    if has_leaf(frame):
        label, confidence = predict_frame(frame)
        pct   = confidence * 100
        text  = f"{label}  {pct:.1f}%"

        # BUG FIX: only colour green when confidence is actually high enough
        color = COLOR_OK if confidence >= CONFIDENCE_THRESHOLD else COLOR_LOW
        _draw_label(frame, text, color)

        # ── thin confidence bar at bottom of frame ──────────────
        bar_w = int(frame.shape[1] * confidence)
        cv2.rectangle(
            frame,
            (0, frame.shape[0] - 6),
            (bar_w, frame.shape[0]),
            color, -1,
        )
    else:
        _draw_label(frame, "Scanning for leaf...", COLOR_SCAN)

    return frame


# ─────────────────────────────────────────────
#  Standalone OpenCV window (run this file directly)
# ─────────────────────────────────────────────
def run_live_detection(camera_index: int = 0) -> None:
    """
    Opens a webcam window and runs live disease detection.
    Press  Q  or  ESC  to quit.
    """
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
                print("[!] Failed to grab frame — retrying …")
                continue

            annotated = process_frame(frame)
            cv2.imshow("Crop Disease Detection  |  Q to quit", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[✓] Camera released.")


if __name__ == "__main__":
    run_live_detection()
