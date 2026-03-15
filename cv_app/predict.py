import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "crop_disease_mobilenet.h5")
CLASS_PATH = os.path.join(os.path.dirname(__file__), "class_names.txt")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = 224
GREEN_THRESHOLD = 0.05


def has_leaf(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])

    return green_ratio > GREEN_THRESHOLD


def predict_frame(frame):
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    predictions = model.predict(img, verbose=0)

    confidence = float(np.max(predictions))

    class_index = int(np.argmax(predictions))

    label = class_names[class_index]

    return label, confidence