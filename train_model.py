"""
train_model.py
STEP 2 of the project: train a neural network classifier on the
hand-landmark dataset collected in data_collection.py.

Run:
    python train_model.py

Input:
    data/landmarks.csv

Output:
    models/sign_model.h5      -> trained Keras model
    models/label_encoder.pkl  -> maps class index back to the letter
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

DATA_PATH = "data/landmarks.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def build_model(input_dim, num_classes):
    """A small, fast dense neural network — well suited to landmark
    features (as opposed to a heavier CNN, which is meant for raw pixels)."""
    # use tensorflow.keras (tf.keras) which is already imported as tf
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not os.path.isfile(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Run data_collection.py first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(input_dim=X.shape[1], num_classes=len(encoder.classes_))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=2,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save(MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Label encoder saved to {ENCODER_PATH}")


if __name__ == "__main__":
    main()
