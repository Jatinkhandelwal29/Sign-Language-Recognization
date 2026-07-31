"""
utils.py
Helper functions shared by data collection, training, and real-time
recognition scripts. Converts Mediapipe hand landmarks into a
normalized feature vector that a machine learning model can learn from.
"""

import numpy as np

# 21 hand landmarks, each with (x, y) -> 42 features per hand
FEATURE_LENGTH = 42


def extract_landmark_features(hand_landmarks):
    """
    Convert a Mediapipe hand_landmarks object into a flat, normalized
    1D feature vector of length 42 (21 landmarks x (x, y)).

    Normalization steps:
    1. Translate all points so the wrist (landmark 0) becomes the origin.
       This makes the features invariant to WHERE the hand is in the frame.
    2. Scale all points by the largest distance from the wrist.
       This makes the features invariant to hand size / distance from camera.

    Returns:
        np.ndarray of shape (42,)
    """
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

    # Step 1: translate relative to the wrist landmark
    wrist = coords[0]
    coords = coords - wrist

    # Step 2: scale so the farthest point from the wrist has distance 1
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist

    return coords.flatten()
