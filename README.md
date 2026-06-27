Crop Disease Detection AI

An AI-powered web application that detects crop leaf diseases using a trained MobileNet deep learning model. The application allows users to upload an image of a crop leaf through a simple web interface and predicts the disease in real time.

The project uses Python, FastAPI, TensorFlow/Keras, OpenCV, HTML, CSS, and JavaScript.

Features
Detect crop leaf diseases from uploaded images
FastAPI backend
Responsive web interface
MobileNet-based deep learning model
Real-time prediction
No internet connection required after setup
Lightweight frontend using HTML, CSS, and JavaScript
Technologies Used
Python
FastAPI
TensorFlow
Keras
OpenCV
NumPy
Pillow
HTML
CSS
JavaScript
Project Structure
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

Note: The model files are not included in this repository because of their large size. Download them separately using the links below.

Installation Guide
Step 1: Clone the repository
git clone https://github.com/deepu8900/crop-disease-detection-ai.git

Go into the project folder.

cd crop-disease-detection-ai
Step 2: Create a Virtual Environment (Recommended)

Windows

python -m venv venv

Activate it.

Command Prompt

venv\Scripts\activate

PowerShell

venv\Scripts\Activate.ps1
Step 3: Install Required Libraries

Install the required packages one by one (or in a single command).

pip install fastapi
pip install uvicorn
pip install tensorflow
pip install keras
pip install numpy
pip install opencv-python
pip install pillow
pip install python-multipart
pip install h5py

Or all at once:

pip install fastapi uvicorn tensorflow keras numpy opencv-python pillow python-multipart h5py
Step 4: Download the AI Models

Download these three model files from the Google Drive links:

crop_disease_mobilenet.h5
leaf_detector.weights.h5
leaf_detector_weights.npy

After downloading, place all three files inside the models folder.

models/
├── crop_disease_mobilenet.h5
├── leaf_detector.weights.h5
└── leaf_detector_weights.npy
Step 5: Run the Application

Open a terminal in the project folder and run:

uvicorn api.main:app --reload

If everything is configured correctly, FastAPI will start the server.

Example output:

INFO: Uvicorn running on http://127.0.0.1:8000

Open the displayed URL in your browser to use the application.

How to Use
Open the application in your browser.
Upload a crop leaf image.
Wait a few seconds while the model processes the image.
View the predicted disease and confidence score (if displayed by your application).
Notes
The dataset used to train the model is not required to run the project.
Only the three trained model files are needed.
Once the project is set up, it works completely offline.
Troubleshooting
ModuleNotFoundError

Install the missing library using:

pip install library_name

Example:

pip install tensorflow
Model file not found

Make sure all three downloaded model files are inside the models folder.

Port already in use

Run the application on another port.

uvicorn api.main:app --reload --port 8001
TensorFlow Installation Error

Upgrade pip before installing TensorFlow.

python -m pip install --upgrade pip

Then reinstall TensorFlow.


Author

Deepu

GitHub:
https://github.com/deepu8900