# Sign Language Recognition — Step-by-Step Guide

This project recognizes hand-sign gestures (e.g., letters A–Z) in real
time from your webcam, using **Mediapipe** for hand tracking and a
**neural network** for classification — matching the project listed
on your resume (Python, OpenCV, TensorFlow/Keras, Mediapipe).

## How it works (big picture)

Instead of feeding raw camera images into a CNN (slow, needs huge
datasets, sensitive to background/lighting), this project:

1. Uses **Mediapipe Hands** to detect 21 key points (landmarks) on
   your hand — fingertips, knuckles, wrist, etc.
2. Converts those 21 (x, y) points into a small **42-number feature
   vector**, normalized so it doesn't matter where your hand is in
   the frame or how close it is to the camera.
3. Trains a lightweight **neural network classifier** to map that
   42-number vector to a letter (A, B, C...).
4. Runs the same pipeline live on webcam video for real-time
   prediction.

This is the same technique used in most production-grade,
lightweight sign-language/gesture apps, and it runs smoothly even
without a GPU.

## Project files

| File | Purpose |
|---|---|
| `utils.py` | Shared function that turns Mediapipe hand landmarks into a normalized feature vector |
| `data_collection.py` | Step 1 — collect your own labeled hand-gesture data from webcam |
| `train_model.py` | Step 2 — train the neural network on the collected data |
| `real_time_recognition.py` | Step 3 — run live webcam recognition using the trained model |
| `requirements.txt` | List of Python packages to install |

## Step 0 — Setup

Install Python 3.9–3.11 (Mediapipe doesn't yet support the very
latest Python versions), then:

```bash
pip install -r requirements.txt
```

This installs: `opencv-python`, `mediapipe`, `numpy`, `pandas`,
`scikit-learn`, `tensorflow`, `joblib`.

## Step 1 — Collect training data

Run:

```bash
python data_collection.py
```

- A webcam window opens.
- Hold up a hand sign (e.g., the letter "A") and **press the "A"
  key** on your keyboard. This both selects "A" as the current label
  and starts recording.
- Keep your hand in frame, moving it slightly (angle, distance,
  lighting) so the model sees variety. Aim for **100–200 samples per
  letter**.
- Press **SPACE** to pause recording (useful when repositioning your
  hand or switching signs), then press the letter key again (or
  SPACE) to resume.
- Move on to the next letter by pressing its key.
- Press **ESC** when you're done collecting all the signs you want.

This creates `data/landmarks.csv`, where each row is 42 landmark
features plus the label you assigned.

**Tip:** Start with a small set (e.g., A, B, C, D, E) to confirm the
whole pipeline works end-to-end before recording all 26 letters.

## Step 2 — Train the model

Run:

```bash
python train_model.py
```

This script:
1. Loads `data/landmarks.csv`.
2. Splits it into training (80%) and testing (20%) sets.
3. Builds a small neural network:
   - Dense(128) → Dropout → Dense(64) → Dropout → Dense(num_classes, softmax)
4. Trains it with early stopping (stops automatically once it stops
   improving, to avoid overfitting).
5. Prints a **classification report** (precision/recall/F1 per
   letter) and a **confusion matrix**, so you can see which signs
   get confused with each other.
6. Saves:
   - `models/sign_model.h5` — the trained network
   - `models/label_encoder.pkl` — the mapping from class index back
     to the actual letter

If accuracy is low, the usual fix is: collect more/varied samples per
letter, or remove signs that look too visually similar for this
approach.

## Step 3 — Real-time recognition

Run:

```bash
python real_time_recognition.py
```

- Opens your webcam.
- Detects your hand, extracts the same 42-feature vector used in
  training, and feeds it to the trained model.
- Displays the predicted letter and confidence percentage on screen
  live.
- If the model isn't confident (below 60%), it shows "?" instead of
  guessing wrong.
- Press **ESC** to quit.

## How this maps to the resume bullet points

- *"Built a hand-gesture recognition system to classify sign
  language alphabets/gestures from webcam video in real time"* →
  `real_time_recognition.py`
- *"Used Mediapipe/OpenCV for hand landmark detection"* → `utils.py`
  + the Mediapipe Hands calls in each script
- *"a CNN classifier to map gestures to corresponding letters"* → in
  this implementation, replaced with a dense neural network trained
  on landmarks, which is faster and more robust for a hobby/resume
  project. If you specifically want a CNN on raw image frames
  instead, see the extension idea below.

## Possible extensions (nice for interviews or a v2)

- **Multi-hand support**: set `max_num_hands=2` in the Mediapipe
  config for two-handed signs.
- **Sequence-based signs (e.g., words, not just letters)**: feed a
  sequence of frames into an LSTM/GRU instead of a single-frame
  dense network.
- **Image-based CNN alternative**: instead of landmarks, crop the
  hand region from each frame, resize to e.g. 64x64, and train a
  small CNN (`Conv2D` → `MaxPooling2D` → ... → `Dense`) directly on
  pixels. More resource-intensive, but closer to a "pure CNN"
  pipeline if that's specifically what you want to describe.
- **Text-to-speech**: speak out the recognized letter/word as it's
  detected, for an accessibility angle.

## Troubleshooting

- **"Could not open webcam"** — another app may be using the camera,
  or you need to grant camera permission to your terminal/IDE.
- **Mediapipe install errors** — make sure you're on Python 3.9–3.11;
  Mediapipe wheels aren't published for every Python version
  immediately.
- **Low accuracy** — collect more samples per class, and make sure
  your hand stays fully in frame during collection.
