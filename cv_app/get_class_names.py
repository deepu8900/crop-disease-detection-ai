import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH="../models/crop_disease_mobilenet.h5"
CLASS_FILE="class_names.txt"
IMG_SIZE=224
CONFIDENCE_THRESHOLD=0.5

model=tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE,"r") as f:
    class_names=[line.strip() for line in f.readlines()]

def preprocess_image(img):
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    return img

def predict_frame(frame):
    processed=preprocess_image(frame)
    predictions=model.predict(processed,verbose=0)
    confidence=np.max(predictions)
    class_index=np.argmax(predictions)
    predicted_class=class_names[class_index]
    return predicted_class,confidence

def detect_from_image():
    image_path=input("Enter full image path: ")
    if not os.path.exists(image_path):
        print("File not found!")
        return
    img=cv2.imread(image_path)
    predicted_class,confidence=predict_frame(img)
    if confidence<CONFIDENCE_THRESHOLD:
        label="No leaf or plant detected"
    else:
        label=f"{predicted_class} ({confidence*100:.2f}%)"
    print("Prediction:",label)
    cv2.putText(img,label,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.imshow("Crop Disease Detection",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_from_webcam():
    cap=cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        predicted_class,confidence=predict_frame(frame)
        if confidence<CONFIDENCE_THRESHOLD:
            label="No leaf detected"
        else:
            label=f"{predicted_class} ({confidence*100:.1f}%)"
        cv2.putText(frame,label,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow("Live Crop Disease Detection",frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("1 -> Image from Desktop")
    print("2 -> Webcam Live Detection")
    choice=input("Enter choice (1/2): ")
    if choice=="1":
        detect_from_image()
    elif choice=="2":
        detect_from_webcam()
    else:
        print("Invalid Choice")

if __name__=="__main__":
    main()