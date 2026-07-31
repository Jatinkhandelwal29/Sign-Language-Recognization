"""
real_time_recognition.py
STEP 3 of the project: real-time sign language recognition using the
webcam, Mediapipe hand landmarks, and the model trained in train_model.py.

Run:
    python real_time_recognition.py

Requires (produced by train_model.py):
    models/sign_model.h5
    models/label_encoder.pkl
"""

import os
import joblib
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

from utils import extract_landmark_features

MODEL_PATH = "models/sign_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.6


def main():
    if not (os.path.isfile(MODEL_PATH) and os.path.isfile(ENCODER_PATH)):
        print("Trained model not found. Run train_model.py first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("=== Real-Time Sign Language Recognition ===")
    print("Press ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        predicted_label = "-"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                features = extract_landmark_features(hand_landmarks).reshape(1, -1)
                probs = model.predict(features, verbose=0)[0]
                class_idx = int(np.argmax(probs))
                confidence = float(probs[class_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_label = encoder.inverse_transform([class_idx])[0]
                else:
                    predicted_label = "?"

        cv2.putText(
            frame,
            f"Sign: {predicted_label} ({confidence * 100:.0f}%)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
