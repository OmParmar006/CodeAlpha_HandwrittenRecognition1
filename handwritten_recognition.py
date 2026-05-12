# ============================================================
# CodeAlpha Machine Learning Internship - Task 3
# Project  : Handwritten Character Recognition
# Dataset  : MNIST (auto-downloaded via Keras)
# Model    : Convolutional Neural Network (CNN)
# Author   : [Your Full Name]
# Date     : May 2026
# GitHub   : github.com/[yourname]/CodeAlpha_HandwrittenRecognition
# ============================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")


# ----------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------

NUM_CLASSES  = 10       # digits 0-9
EPOCHS       = 20
BATCH_SIZE   = 64
RANDOM_STATE = 42

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("=" * 55)
print(" CodeAlpha - Handwritten Character Recognition")
print("=" * 55)


# ----------------------------------------------------------
# 2. Load Dataset
# ----------------------------------------------------------

print("\n[1/6] Loading MNIST dataset...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"    Train samples : {X_train.shape[0]}")
print(f"    Test  samples : {X_test.shape[0]}")
print(f"    Image size    : {X_train.shape[1]}x{X_train.shape[2]} pixels")
print(f"    Classes       : {NUM_CLASSES} (digits 0-9)")


# ----------------------------------------------------------
# 3. EDA - Visualize samples
# ----------------------------------------------------------

print("\n[2/6] Generating EDA plots...")

# sample images
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for digit in range(10):
    indices = np.where(y_train == digit)[0][:3]
    for row, idx in enumerate(indices):
        axes[row, digit].imshow(X_train[idx], cmap="gray")
        axes[row, digit].axis("off")
        if row == 0:
            axes[row, digit].set_title(str(digit), fontsize=12)

plt.suptitle("Sample Images per Digit (MNIST)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("sample_digits.png", dpi=150)
plt.close()
print("    Saved: sample_digits.png")

# class distribution
plt.figure(figsize=(8, 4))
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts, color="steelblue", edgecolor="black")
plt.title("Class Distribution - Training Set")
plt.xlabel("Digit")
plt.ylabel("Number of Samples")
plt.xticks(range(10))
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()
print("    Saved: class_distribution.png")


# ----------------------------------------------------------
# 4. Preprocess Data
# ----------------------------------------------------------

print("\n[3/6] Preprocessing data...")

# normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

# reshape for CNN: (samples, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# one-hot encode labels
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

print(f"    X_train shape : {X_train.shape}")
print(f"    X_test  shape : {X_test.shape}")
print(f"    Pixel range   : {X_train.min():.1f} to {X_train.max():.1f}")


# ----------------------------------------------------------
# 5. Build CNN Model
# ----------------------------------------------------------

print("\n[4/6] Building CNN model...")

model = Sequential([
    # first conv block
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # second conv block
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # third conv block
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Dropout(0.25),

    # classifier
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

total_params = model.count_params()
print(f"\n    Total parameters: {total_params:,}")


# ----------------------------------------------------------
# 6. Train Model
# ----------------------------------------------------------

print("\n[5/6] Training model...")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()
print("    Saved: training_curves.png")


# ----------------------------------------------------------
# 7. Evaluate
# ----------------------------------------------------------

print("\n[6/6] Evaluating model...\n")

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"    Test Accuracy : {test_acc * 100:.2f}%")
print(f"    Test Loss     : {test_loss:.4f}")

y_pred     = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true     = y_test

print("\n    Classification Report:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)
plt.title("Confusion Matrix - Handwritten Digit Recognition")
plt.xlabel("Predicted Digit")
plt.ylabel("True Digit")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("    Saved: confusion_matrix.png")

# visualize predictions (correct and wrong)
fig, axes = plt.subplots(2, 10, figsize=(16, 4))

wrong_idx   = np.where(y_pred != y_true)[0]
correct_idx = np.where(y_pred == y_true)[0]

for i in range(10):
    # correct predictions (top row)
    idx = correct_idx[i]
    axes[0, i].imshow(X_test[idx].reshape(28, 28), cmap="gray")
    axes[0, i].set_title(f"✓ {y_pred[idx]}", color="green", fontsize=9)
    axes[0, i].axis("off")

    # wrong predictions (bottom row)
    idx = wrong_idx[i]
    axes[1, i].imshow(X_test[idx].reshape(28, 28), cmap="gray")
    axes[1, i].set_title(f"P:{y_pred[idx]} T:{y_true[idx]}", color="red", fontsize=9)
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Correct", fontsize=10)
axes[1, 0].set_ylabel("Wrong", fontsize=10)

plt.suptitle("Correct vs Wrong Predictions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("predictions_sample.png", dpi=150)
plt.close()
print("    Saved: predictions_sample.png")

# save model
model.save("handwritten_recognition_model.h5")
print("    Saved: handwritten_recognition_model.h5")

print("\n" + "=" * 55)
print(f" Final Test Accuracy : {test_acc * 100:.2f}%")
print(" Task 3 completed successfully.")
print("=" * 55)