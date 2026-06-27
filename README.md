# Crop Disease Detection AI

An AI-powered web application that detects crop leaf diseases using a trained MobileNet deep learning model. The application allows users to upload an image of a crop leaf through a simple web interface and predicts the disease in real time.

**Tech Stack:** Python • FastAPI • TensorFlow/Keras • OpenCV • HTML • CSS • JavaScript

---

# Features

- Detect crop leaf diseases from uploaded images
- FastAPI backend
- Responsive web interface
- MobileNet-based deep learning model
- Real-time prediction
- Works offline after setup
- Lightweight frontend using HTML, CSS and JavaScript

---

# Technologies Used

- Python
- FastAPI
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pillow
- HTML
- CSS
- JavaScript

---

# Project Structure

```text
crop-disease-detection-ai/
│
├── api/
│   └── main.py
│
├── cv_app/
│   ├── get_classes_names.py
│   └── live_detection.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
└── models/
    ├── crop_disease_mobilenet.h5
    ├── leaf_detector.weights.h5
    └── leaf_detector_weights.npy
```

> **Note:** The model files are **not included** in this repository because of their large size. Download them separately using the Google Drive links.

---

# Installation Guide

## Step 1: Clone the Repository

```bash
git clone https://github.com/deepu8900/crop-disease-detection-ai.git
```

Move into the project directory.

```bash
cd crop-disease-detection-ai
```

---

## Step 2: Create a Virtual Environment (Recommended)

Create a virtual environment:

```bash
python -m venv venv
```

### Activate the Virtual Environment

**Command Prompt**

```bash
venv\Scripts\activate
```

**PowerShell**

```powershell
venv\Scripts\Activate.ps1
```

---

## Step 3: Install Required Libraries

Install all required Python libraries using the following command:

```bash
pip install fastapi uvicorn tensorflow keras numpy opencv-python pillow python-multipart h5py
```

---

## Step 4: Download the AI Models

The trained model files are **not included** in this repository because they exceed GitHub's file size limits.

Download the following files from Google Drive and place them inside the `models` folder.

| Model File | Download Link |
|------------|---------------|
| `crop_disease_mobilenet.h5` | https://drive.google.com/file/d/1kE0CHegc_DnpKIUq7eMASo-B_Hjo3jK0/view?usp=drive_link |
| `leaf_detector.weights.h5` | https://drive.google.com/file/d/1siwgizbkDo6zAEcukb0K2JxM07-HkZF0/view?usp=drive_link |
| `leaf_detector_weights.npy` | https://drive.google.com/file/d/1wL08qITkUJI_ZaPEQt-28xq029JtFj9p/view?usp=drive_link |

After downloading, create a folder named `models` (if it does not already exist) in the project root and place the files as shown below.

```text
models/
├── crop_disease_mobilenet.h5
├── leaf_detector.weights.h5
└── leaf_detector_weights.npy
```
---

## Step 5: Run the Application

Start the FastAPI server by running:

```bash
uvicorn api.main:app --reload
```

If everything is configured correctly, you should see:

```text
INFO: Uvicorn running on http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000` in your web browser to use the application.

---

# How to Use

1. Launch the application in your browser.
2. Upload a crop leaf image.
3. Wait a few seconds while the AI model processes the image.
4. View the predicted crop disease.

---

# Notes

- The dataset used to train the model is **not required** to run this project.
- Only the three trained model files are needed.
- The application works completely offline after setup.
- Ensure all model files are placed in the `models` directory before starting the server.

---

# Troubleshooting

## ModuleNotFoundError

Install the missing package using:

```bash
pip install <library_name>
```

Example:

```bash
pip install tensorflow
```

---

## Model File Not Found

Ensure the following files are present inside the `models` folder:

- `crop_disease_mobilenet.h5`
- `leaf_detector.weights.h5`
- `leaf_detector_weights.npy`

---

## Port Already in Use

Run the application on a different port:

```bash
uvicorn api.main:app --reload --port 8001
```

---

## TensorFlow Installation Error

Upgrade `pip` before installing TensorFlow:

```bash
python -m pip install --upgrade pip
```

Then reinstall TensorFlow:

```bash
pip install tensorflow
```

---

# Author

**Deepu**

GitHub: [@deepu8900](https://github.com/deepu8900)
