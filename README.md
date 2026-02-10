# Face-recognition-system-
Face Recognition System using OpenCV

This project implements a real-time face recognition system using OpenCV. The system detects human faces from a webcam feed and identifies individuals based on a trained dataset using the Local Binary Pattern Histogram (LBPH) recognition algorithm.

Features

Real-time face detection using Haar Cascade classifiers

Dataset generation through webcam capture

Face recognition using the LBPH algorithm

Automatic labeling of recognized individuals

Unknown face detection based on confidence threshold

Project Structure
collect_faces.py        # Dataset collection using webcam
train_model.py          # Training the LBPH face recognition model
recognize_faces.py      # Real-time recognition using trained model
haarcascade_frontalface_default.xml   # Haar Cascade face detector

Workflow

Capture face samples using collect_faces.py to create labeled datasets.

Train the recognition model using train_model.py.

Perform real-time recognition using recognize_faces.py.

Requirements

Python 3.x

OpenCV (opencv-contrib-python)

NumPy

Install dependencies:

pip install opencv-contrib-python numpy

Running the Project

Dataset collection:

python collect_faces.py


Model training:

python train_model.py


Real-time recognition:

python recognize_faces.py

Description

The system uses Haar Cascade classifiers for face detection and the LBPH algorithm for recognition. Facial images are converted to grayscale, resized to a fixed dimension, and used to train the recognition model. During runtime, detected faces are compared with trained features to identify individuals or label them as unknown.
