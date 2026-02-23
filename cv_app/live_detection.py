import cv2
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tkinter as tk
from tkinter import filedialog

MODEL_PATH=os.path.join(os.path.dirname(__file__),"..","models","crop_disease_mobilenet.h5")
CLASS_PATH=os.path.join(os.path.dirname(__file__),"class_names.txt")

print("Loading model...")
model=tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

with open(CLASS_PATH,"r") as f:
    class_names=[line.strip() for line in f.readlines()]

def format_class_name(name):
    if "___" in name:
        plant,disease=name.split("___")
        plant=plant.replace("_"," ").replace(",","").replace("(including_sour)","").replace("(maize)","").replace("(","").replace(")","").strip()
        disease=disease.replace("_"," ").replace("(","").replace(")","").strip()
        if plant.lower() in disease.lower():
            disease=disease.lower().replace(plant.lower(),"").strip()
        return f"{plant.title()} {disease.title()}".strip()
    return name.replace("_"," ").title()

IMG_SIZE=224
CONFIDENCE_THRESHOLD=0.75
GREEN_THRESHOLD=0.05

def has_leaf(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_green=np.array([25,40,40])
    upper_green=np.array([85,255,255])
    mask=cv2.inRange(hsv,lower_green,upper_green)
    green_ratio=np.sum(mask>0)/(frame.shape[0]*frame.shape[1])
    return green_ratio>GREEN_THRESHOLD

def predict_frame(frame):
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    img=cv2.resize(frame,(IMG_SIZE,IMG_SIZE))
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    predictions=model.predict(img,verbose=0)
    confidence=float(np.max(predictions))
    class_index=int(np.argmax(predictions))
    raw_name=class_names[class_index]
    clean_name=format_class_name(raw_name)
    return clean_name,confidence
def draw_text(frame,text1,text2,color):
    font=cv2.FONT_HERSHEY_SIMPLEX
    thickness=2
    font_scale=0.7

    (w1,h1),_=cv2.getTextSize(text1,font,font_scale,thickness)
    (w2,h2),_=cv2.getTextSize(text2,font,font_scale,thickness)

    x=20
    y1=40
    y2=y1+h1+15

    max_width=max(w1,w2)

    cv2.rectangle(frame,
                  (x-10,y1-h1-10),
                  (x+max_width+10,y2+10),
                  (0,0,0),
                  -1)

    cv2.putText(frame,text1,(x,y1),font,font_scale,color,thickness,cv2.LINE_AA)
    cv2.putText(frame,text2,(x,y2),font,font_scale,color,thickness,cv2.LINE_AA)


root=tk.Tk()
root.withdraw()

print("Choose input method:")
print("1 - Webcam Live Detection")
print("2 - Select Image File")

choice=input("Enter 1 or 2: ")
from collections import deque
prediction_buffer = deque(maxlen=5)

if choice=="1":
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Press 'q' to quit.")

    while True:
        ret,frame=cap.read()
        if not ret:
            break

        if has_leaf(frame):
            label,confidence=predict_frame(frame)
            prediction_buffer.append((label,confidence))

            avg_confidence = sum(c for _,c in prediction_buffer)/len(prediction_buffer)
            label = max(set(l for l,_ in prediction_buffer), key=lambda x: sum(1 for l,_ in prediction_buffer if l==x))
            confidence = avg_confidence
            if confidence>CONFIDENCE_THRESHOLD:
                text1=f"Disease: {label}"
                text2=f"Confidence: {confidence*100:.2f}%"
                color=(0,255,0)
            else:
                text="Leaf detected but unsure"
                color=(0,165,255)
        else:
            text="No leaf detected"
            color=(0,0,255)

        draw_text(frame,text,color)
        cv2.imshow("Crop Disease Detection",frame)

        key=cv2.waitKey(1)
        if key==ord('q') or key==27:
            break

    cap.release()
    cv2.destroyAllWindows()

elif choice=="2":

    file_path=filedialog.askopenfilename(
        title="Select Leaf Image",
        filetypes=[("Image Files","*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print("No file selected.")
        exit()

    print("Processing image...")

    frame=cv2.imread(file_path)

    if has_leaf(frame):
        label,confidence=predict_frame(frame)
        if confidence>CONFIDENCE_THRESHOLD:
            text1=f"Disease: {label}"
            text2=f"Confidence: {confidence*100:.2f}%"
            color=(0,255,0)
        else:
            text="Leaf detected but unsure"
            color=(0,165,255)
    else:
        text="No leaf detected"
        color=(0,0,255)
    draw_text(frame,text1,text2,color)

    cv2.imshow("Prediction Result",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Invalid choice.")

root.destroy()
