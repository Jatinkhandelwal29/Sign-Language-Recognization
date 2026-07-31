"""
data_collection.py
STEP 1 of the project: collect labeled hand-landmark data from your
webcam to build a training dataset for sign language gestures.

Run:
    python data_collection.py

Controls (while the webcam window is focused):
    - Press a letter key (A-Z)  -> selects that letter as the current
                                    label AND starts recording samples.
    - Press SPACE               -> pause / resume recording.
    - Press ESC                 -> quit and save everything to CSV.

Output:
    data/landmarks.csv
    Each row = [42 landmark features..., label]

Tip: For each letter, collect at least 100-200 samples, moving your
hand slightly (angle, distance, lighting) so the model generalizes well.
"""

import os
import csv
import cv2
import mediapipe as mp

from utils import extract_landmark_features, FEATURE_LENGTH

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    file_exists = os.path.isfile(CSV_PATH)
    csv_file = open(CSV_PATH, mode="a", newline="")
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        header = [f"f{i}" for i in range(FEATURE_LENGTH)] + ["label"]
        csv_writer.writerow(header)

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
        print("Could not open webcam. Check your camera connection/permissions.")
        return

    current_label = None
    recording = False
    sample_count = 0

    print("=== Sign Language Data Collection ===")
    print("Press a letter key (A-Z) to select a label and start recording.")
    print("Press SPACE to pause/resume. Press ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                if recording and current_label is not None:
                    features = extract_landmark_features(hand_landmarks)
                    csv_writer.writerow(list(features) + [current_label])
                    sample_count += 1

        status = (
            f"Label: {current_label or '-'} | "
            f"Recording: {recording} | Samples this label: {sample_count}"
        )
        cv2.putText(
            frame, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow("Data Collection - Sign Language", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            recording = not recording
        elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
            current_label = chr(key).upper()
            recording = True
            sample_count = 0

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\nData saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
