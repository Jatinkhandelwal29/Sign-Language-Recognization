# Sign Language Recognition

Real-time hand-sign recognition using **MediaPipe** hand-landmark tracking and a **neural network classifier**, built with Python, OpenCV, and TensorFlow/Keras.

## Overview

This project detects a hand in a webcam feed, extracts 21 hand landmarks with MediaPipe, converts them into a normalized 42-value feature vector, and classifies the gesture (e.g., a sign-language letter A–Z) using a trained neural network — all in real time.

Instead of feeding raw camera frames into a CNN (slow, data-hungry, sensitive to background/lighting), the pipeline works on lightweight landmark coordinates, which makes it fast enough to run smoothly without a GPU.

## Features

- Custom webcam-based data collection tool for building your own gesture dataset
- Landmark-based features (position- and scale-invariant) instead of raw pixels — fast and robust
- Dense neural network classifier (128 → 64 → softmax) with dropout and early stopping
- Prints a classification report and confusion matrix after training
- Live webcam inference with on-screen prediction + confidence score
- Unknown/low-confidence gestures are shown as `?` instead of a wrong guess

## Tech Stack

Python · OpenCV · MediaPipe · TensorFlow/Keras · NumPy · Pandas · Scikit-learn

## Project Structure

```
Sign-Language-Recognization/
├── utils.py                   # Landmark → normalized feature vector
├── data_collection.py         # Step 1: record labeled gesture data from webcam
├── train_model.py             # Step 2: train the neural network classifier
├── real_time_recognition.py   # Step 3: live webcam recognition
├── requirements.txt
├── PROJECT_GUIDE.md           # Full step-by-step walkthrough
├── data/
│   └── landmarks.csv          # Collected landmark data (generated)
└── models/                    # Trained model + label encoder (generated)
```

## Installation

```bash
git clone https://github.com/Jatinkhandelwal29/Sign-Language-Recognization.git
cd Sign-Language-Recognization
pip install -r requirements.txt
```

Requirements:
- Python 3.9–3.11 (MediaPipe doesn't yet support the very latest Python versions)
- A working webcam

## Usage

```bash
# 1. Collect your own gesture data
#    Press a letter key (A-Z) to select it as the label and start recording.
#    Press SPACE to pause/resume. Press ESC to stop and save.
python data_collection.py

# 2. Train the classifier on the collected data
python train_model.py

# 3. Run real-time recognition
python real_time_recognition.py
```

Each step depends on the output of the one before it:

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `data_collection.py` | Webcam | `data/landmarks.csv` |
| 2 | `train_model.py` | `data/landmarks.csv` | `models/sign_model.h5`, `models/label_encoder.pkl` |
| 3 | `real_time_recognition.py` | Webcam + trained model | Live predictions on screen |

See `PROJECT_GUIDE.md` for a more detailed walkthrough of every stage, controls, and troubleshooting tips (including possible extensions like two-hand support, word-level LSTM sequences, or a pure image-based CNN).

## How It Works

1. **MediaPipe Hands** detects 21 key points on the hand (fingertips, knuckles, wrist, etc.) per frame.
2. `utils.py` converts those 21 `(x, y)` points into a normalized 42-value feature vector:
   - Translates all points so the wrist becomes the origin (position-invariant)
   - Scales by the farthest point from the wrist (size/distance-invariant)
3. `train_model.py` trains a small dense neural network to map that 42-value vector to a letter.
4. `real_time_recognition.py` runs the same pipeline live on webcam video, showing the predicted letter and confidence score.

## Troubleshooting

- **"Could not open webcam"** — another app may be using the camera, or your terminal/IDE needs camera permission.
- **MediaPipe install errors** — make sure you're on Python 3.9–3.11.
- **Low accuracy** — collect more/varied samples per letter, and keep your hand fully in frame during data collection.

## Author

**Jatin Khandelwal**
