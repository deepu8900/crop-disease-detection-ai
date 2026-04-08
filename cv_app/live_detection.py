import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
import threading

ROOT_DIR              = Path(__file__).resolve().parent.parent
MODEL_PATH            = ROOT_DIR / "models" / "crop_disease_mobilenet.h5"
NAMES_PATH            = ROOT_DIR / "cv_app"  / "class_names.txt"
LEAF_DETECTOR_WEIGHTS_H5  = ROOT_DIR / "models" / "leaf_detector.weights.h5"
LEAF_DETECTOR_WEIGHTS_NPY = ROOT_DIR / "models" / "leaf_detector_weights.npy"

IMG_SIZE             = 224
CONFIDENCE_THRESHOLD = 0.75

GREEN_RANGES = [
    (np.array([22,  40,  40], dtype=np.uint8),
     np.array([95, 255, 255], dtype=np.uint8)),  
    (np.array([15,  40,  40], dtype=np.uint8),
     np.array([22, 255, 255], dtype=np.uint8)),   
]
LEAF_PIXEL_MIN = 0.06

LEAF_CLASS_INDEX   = 0      
LEAF_DETECT_THRESH = 0.70   

COLOR_OK   = (0, 220, 80)
COLOR_LOW  = (0, 180, 255)
COLOR_SCAN = (200, 200, 0)
COLOR_BG   = (0, 0, 0)


def _build_disease_model(num_classes: int) -> tf.keras.Model:
    base    = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None)
    x       = base.output
    x       = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
    x       = tf.keras.layers.Dropout(0.3, name='dropout')(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax',
        name='predictions_39' if num_classes == 39 else 'predictions')(x)
    return tf.keras.Model(inputs=base.input, outputs=outputs)

print(f"[*] Loading disease model from {MODEL_PATH} …")
try:
    model          = tf.keras.models.load_model(str(MODEL_PATH))
    NUM_CLASSES    = model.output_shape[-1]
    print(f"[✓] Disease model ready. ({NUM_CLASSES} classes)")
except Exception:
    import h5py
    with h5py.File(str(MODEL_PATH), 'r') as f:
        layers      = list(f['model_weights'].keys())
        NUM_CLASSES = 39 if 'predictions_39' in layers else 38
    print(f"[*] Rebuilding architecture ({NUM_CLASSES} classes) …")
    model = _build_disease_model(NUM_CLASSES)
    model.build((None, 224, 224, 3))
    model.load_weights(str(MODEL_PATH), by_name=True, skip_mismatch=True)
    print(f"[✓] Disease model ready. ({NUM_CLASSES} classes)")

BG_CLASS_IDX = 38 if NUM_CLASSES == 39 else -1

def _build_leaf_detector() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None)
    return tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ], name='leaf_detector')

leaf_detector = None

if LEAF_DETECTOR_WEIGHTS_NPY.exists():
    
    print(f"[*] Loading leaf detector from {LEAF_DETECTOR_WEIGHTS_NPY} …")
    try:
        leaf_detector = _build_leaf_detector()
        leaf_detector.build((None, 224, 224, 3))
        weights = np.load(str(LEAF_DETECTOR_WEIGHTS_NPY), allow_pickle=True)
        leaf_detector.set_weights(weights)
        print("[✓] Leaf detector ready. (numpy weights)")
    except Exception as e:
        leaf_detector = None
        print(f"[!] Leaf detector numpy load failed: {e}")

elif LEAF_DETECTOR_WEIGHTS_H5.exists():
    
    print(f"[*] Loading leaf detector from {LEAF_DETECTOR_WEIGHTS_H5} …")
    try:
        import h5py
        leaf_detector = _build_leaf_detector()
        leaf_detector.build((None, 224, 224, 3))
        
        weights = []
        with h5py.File(str(LEAF_DETECTOR_WEIGHTS_H5), 'r') as f:
            def collect_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    weights.append(obj[()])
            f.visititems(collect_weights)
        if len(weights) == len(leaf_detector.get_weights()):
            leaf_detector.set_weights(weights)
            print("[✓] Leaf detector ready. (h5 manual load)")
        else:
            raise ValueError(
                f"Weight count mismatch: got {len(weights)}, "
                f"expected {len(leaf_detector.get_weights())}")
    except Exception as e:
        leaf_detector = None
        print(f"[!] Leaf detector h5 load failed: {e}")
        print("    → In Colab run: np.save('leaf_detector_weights.npy', model.get_weights(), allow_pickle=True)")
        print("    → Place leaf_detector_weights.npy in models/")

else:
    print("[!] Leaf detector not found — using green colour check only.")
    print("    → In Colab: np.save('/content/leaf_detector_weights.npy', model.get_weights(), allow_pickle=True)")
    print("    → Place in models/leaf_detector_weights.npy")

if NAMES_PATH.exists():
    with open(NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"[✓] Loaded {len(class_names)} class names.")
else:
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
    print(f"[!] class_names.txt not found — using built-in list ({len(class_names)} classes).")


_lock = threading.Lock()

best_result: dict = {"disease": None, "confidence": 0.0, "snapshot": None}

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

def has_leaf_colour(frame: np.ndarray) -> bool:
    """Green HSV check — used for file upload."""
    hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in GREEN_RANGES:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))
    return np.count_nonzero(combined) / combined.size >= LEAF_PIXEL_MIN


def has_leaf_yolo(frame: np.ndarray) -> bool:
 
    if not has_leaf_colour(frame):
        return False

    if leaf_detector is None:
        return True   

    img      = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img      = (img.astype(np.float32) / 127.5) - 1.0
    img      = np.expand_dims(img, axis=0)
    prob     = float(leaf_detector.predict(img, verbose=0)[0][0])
    leaf_prob = prob if LEAF_CLASS_INDEX == 0 else (1.0 - prob)
    return leaf_prob >= LEAF_DETECT_THRESH

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    img      = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    img      = np.expand_dims(img.astype(np.float32), axis=0)
    return preprocess_input(img)


def predict_frame(frame: np.ndarray) -> tuple[str, float]:
    
    preds      = model.predict(preprocess_frame(frame), verbose=0)[0]
    idx        = int(np.argmax(preds))
    confidence = float(preds[idx])

    # 39-class model — reject background predictions
    if BG_CLASS_IDX >= 0 and idx == BG_CLASS_IDX:
        return "background", confidence

    if confidence < CONFIDENCE_THRESHOLD:
        label = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        return f"Uncertain ({label})", confidence

    label = class_names[idx] if idx < len(class_names) else f"Class {idx}"
    return label, confidence

def _draw_label(frame: np.ndarray, text: str, color: tuple) -> None:
    font             = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.65, 2
    (tw, th), _      = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (15, 15), (25 + tw, 25 + th), COLOR_BG, -1)
    cv2.putText(frame, text, (20, th + 18), font, scale, color, thickness)


def _annotate(frame: np.ndarray, label: str, confidence: float) -> None:
    pct   = confidence * 100
    text  = f"{label}  {pct:.1f}%"
    color = COLOR_OK if confidence >= CONFIDENCE_THRESHOLD else COLOR_LOW
    _draw_label(frame, text, color)
    # confidence bar at bottom
    bar_w = int(frame.shape[1] * confidence)
    cv2.rectangle(frame, (0, frame.shape[0]-6), (bar_w, frame.shape[0]), color, -1)
    # update best result
    with _lock:
        current_best = best_result["confidence"]
    if pct > current_best and confidence >= CONFIDENCE_THRESHOLD:
        update_best_result(label, confidence, frame.copy())


def predict_and_annotate(frame: np.ndarray) -> tuple[str, float, np.ndarray]:
 
    if not has_leaf_yolo(frame):
        return "Scanning for leaf...", COLOR_SCAN

    label, confidence = predict_frame(frame)

    if label == "background":
        return "Scanning for leaf...", COLOR_SCAN

    pct   = confidence * 100
    text  = f"{label}  {pct:.1f}%"
    color = COLOR_OK if confidence >= CONFIDENCE_THRESHOLD else COLOR_LOW

    # update best result
    with _lock:
        current_best = best_result["confidence"]
    if pct > current_best and confidence >= CONFIDENCE_THRESHOLD:
        update_best_result(label, confidence, frame.copy())

    return text, color

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
            text, color = predict_and_annotate(frame)
            _draw_label(frame, text, color)
            cv2.imshow("Crop Disease Detection  |  Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[✓] Camera released.")


if __name__ == "__main__":
    run_live_detection()
