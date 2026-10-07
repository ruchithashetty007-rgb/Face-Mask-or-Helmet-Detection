import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "helmet_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

image_path = input("Enter the full path of the helmet image: ").strip()

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image could not be loaded.")
    exit()

image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32)
image = np.expand_dims(image, axis=0)

image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

prediction = model.predict(image, verbose=0)[0][0]

print("\nRaw model prediction:", prediction)

if prediction >= 0.5:
    print("Current mapping: HELMET")
else:
    print("Current mapping: NO HELMET")