import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Helmet Detection",
    page_icon="🪖",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main {
        background-color: #0b1220;
    }

    .title {
        text-align: center;
        color: #00e5ff;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #b8c4d6;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .card {
        background-color: #151f32;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
    }

    .safe {
        background-color: #123d2b;
        color: #42f59b;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }

    .warning {
        background-color: #4a1f25;
        color: #ff5c5c;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }

    .metric {
        text-align: center;
        background-color: #151f32;
        padding: 20px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "models" / "helmet_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">🪖 Smart Helmet Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered real-time helmet compliance detection using MobileNetV2</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Control Panel")

input_mode = st.sidebar.radio(
    "Select Input",
    ["Upload Image", "Webcam"]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This system uses Transfer Learning with MobileNetV2 "
    "to classify helmet and no-helmet images."
)

st.sidebar.markdown("### Model")
st.sidebar.write("MobileNetV2")
st.sidebar.write("Image Size: 224 × 224")
st.sidebar.write("Test Accuracy: 68.70%")

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image):

    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    prediction = model.predict(image, verbose=0)[0][0]

    # Mapping used in the webcam version
    if prediction >= 0.5:
        label = "NO HELMET"
        confidence = prediction
    else:
        label = "HELMET"
        confidence = 1 - prediction

    return label, confidence * 100


# -----------------------------
# Upload Image
# -----------------------------
if input_mode == "Upload Image":

    st.markdown("## 📷 Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose a helmet image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        image_array = np.array(image)

        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(
                image_array,
                cv2.COLOR_GRAY2RGB
            )

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        label, confidence = predict_image(image_bgr)

        with col2:

            st.markdown("### Detection Result")

            if label == "HELMET":

                st.markdown(
                    '<div class="safe">🟢 SAFE<br>HELMET DETECTED</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="warning">🔴 WARNING<br>NO HELMET DETECTED</div>',
                    unsafe_allow_html=True
                )

            st.metric(
                "Confidence",
                f"{confidence:.1f}%"
            )

# -----------------------------
# Webcam
# -----------------------------
else:

    st.markdown("## 🎥 Live Webcam Detection")

    st.info(
        "Allow camera access when your browser asks for permission."
    )

    camera_image = st.camera_input(
        "Take a picture using your webcam"
    )

    if camera_image is not None:

        image = Image.open(camera_image)

        image_array = np.array(image)

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        label, confidence = predict_image(image_bgr)

        st.markdown("### Detection Result")

        if label == "HELMET":

            st.markdown(
                '<div class="safe">🟢 SAFE<br>HELMET DETECTED</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="warning">🔴 WARNING<br>NO HELMET DETECTED</div>',
                unsafe_allow_html=True
            )

        st.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
    "<center>Developed using Python • TensorFlow • OpenCV • Streamlit • MobileNetV2</center>",
    unsafe_allow_html=True
)