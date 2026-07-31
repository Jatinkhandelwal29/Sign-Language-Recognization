# Sign Language Recognition

Real-time hand-sign recognition using **Mediapipe** hand-landmark
tracking and a **neural network classifier**, built with Python,
OpenCV, and TensorFlow/Keras.

## Overview

This project detects a hand in a webcam feed, extracts 21 hand
landmarks with Mediapipe, converts them into a normalized feature
vector, and classifies the gesture (e.g., a sign-language letter)
using a trained neural network — all in real time.

## Features

- Custom webcam-based data collection tool for building your own gesture dataset
- Landmark-based features (position/scale invariant) instead of raw pixels — fast and robust
- Dense neural network classifier with dropout and early stopping
- Live webcam inference with on-screen prediction + confidence score

## Tech Stack

Python · OpenCV · Mediapipe · TensorFlow/Keras · NumPy · Pandas · Scikit-learn

## Project Structure

```
sign_language_recognition/
├── utils.py                   # Landmark feature extraction
├── data_collection.py         # Step 1: record labeled gesture data
├── train_model.py             # Step 2: train the classifier
├── real_time_recognition.py   # Step 3: live webcam recognition
├── requirements.txt
├── PROJECT_GUIDE.md           # Full step-by-step walkthrough
├── data/                      # Collected landmark data (generated)
└── models/                    # Trained model + label encoder (generated)
```

## Installation

```bash
git clone <your-repo-url>
cd sign_language_recognition
pip install -r requirements.txt
```

## Usage

```bash
# 1. Collect your own gesture data (press A-Z to label, ESC to save)
python data_collection.py

# 2. Train the classifier on the collected data
python train_model.py

# 3. Run real-time recognition
python real_time_recognition.py
```

See `PROJECT_GUIDE.md` for a detailed, step-by-step explanation of
every stage, controls, and troubleshooting tips.

## Requirements

- A working webcam
- Python 3.9–3.11 (for Mediapipe compatibility)

## Author

**Jatin Khandelwal**
