import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# =========================
# SETTINGS
# =========================
IMG_SIZE = (224, 224)

MODEL_PATH = Path(__file__).parent / "models" / "helmet_model.keras"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Starting webcam...")
print("Press Q to quit.")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    print("Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1).")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Resize frame for MobileNetV2
    image = cv2.resize(frame, IMG_SIZE)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare image
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    # Prediction
    prediction = model.predict(image, verbose=0)[0][0]

    # Our classes are:
    # 0 = helmet
    # 1 = no_helmet

    if prediction >= 0.5:
        label = "HELMET"
        confidence = prediction
    else:
        label = "NO HELMET"
        confidence = 1 - prediction

    confidence_percent = confidence * 100

    # Safety status
    if label == "HELMET":
        status = "SAFE"
    else:
        status = "WARNING"

    # Display prediction
    cv2.putText(
        frame,
        f"Prediction: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0) if label == "HELMET" else (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence_percent:.1f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if status == "SAFE" else (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to exit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Show webcam
    cv2.imshow("Smart Helmet Detection - MobileNetV2", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()