import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Helmet Detection",
    page_icon="🪖",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

BASE_DIR = Path(__file__).parent

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "helmet_model.keras"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎛️ Control Panel")

    st.write(
        "This application uses MobileNetV2 "
        "to detect whether a helmet is present."
    )

    st.divider()

    st.subheader("🪖 Classes")

    st.write("🟢 0 → Helmet")
    st.write("🔴 1 → No Helmet")

    st.divider()

    st.subheader("🤖 Model")

    st.write("MobileNetV2")
    st.write("Input Size: 224 × 224")
    st.write("Classes: 2")

    st.divider()

    st.info(
        "Upload an image of a person or rider "
        "to check helmet compliance."
    )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🪖 Smart Helmet Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered helmet compliance detection using MobileNetV2'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHECK MODEL
# ============================================================

if model is None:

    st.error(
        "❌ Model file not found.\n\n"
        "Please make sure this file exists:\n\n"
        "models/helmet_model.keras"
    )

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Choose a helmet image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(
        IMG_SIZE
    )

    # Convert to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    image_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(
            image_array
        )
    )

    # Model prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # ========================================================
    # IMPORTANT CLASS MAPPING
    #
    # 0 = helmet
    # 1 = no_helmet
    #
    # Sigmoid output represents probability of class 1.
    # ========================================================

    no_helmet_probability = float(
        prediction
    )

    helmet_probability = (
        1.0 -
        no_helmet_probability
    )

    # Determine result
    if no_helmet_probability >= 0.5:

        label = "NO HELMET DETECTED"

        confidence = (
            no_helmet_probability
        )

        result_type = "danger"

    else:

        label = "HELMET DETECTED"

        confidence = (
            helmet_probability
        )

        result_type = "safe"

    return (
        label,
        confidence,
        result_type,
        helmet_probability,
        no_helmet_probability
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.divider()

    # ========================================================
    # TWO COLUMNS
    # ========================================================

    image_column, result_column = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # IMAGE
    # ========================================================

    with image_column:

        st.subheader(
            "🖼️ Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    (
        label,
        confidence,
        result_type,
        helmet_probability,
        no_helmet_probability
    ) = predict_image(
        image
    )


    # ========================================================
    # RESULT
    # ========================================================

    with result_column:

        st.subheader(
            "🔍 Detection Result"
        )


        # ----------------------------------------------------
        # HELMET
        # ----------------------------------------------------

        if result_type == "safe":

            st.success(
                "🟢 SAFE\n\n"
                "🪖 HELMET DETECTED"
            )


        # ----------------------------------------------------
        # NO HELMET
        # ----------------------------------------------------

        else:

            st.error(
                "🔴 WARNING\n\n"
                "⚠️ NO HELMET DETECTED"
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence * 100:.1f}%"
        )


        st.divider()


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.subheader(
            "📊 Prediction Probability"
        )

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "🪖 Helmet Probability",
                f"{helmet_probability * 100:.1f}%"
            )


        with col2:

            st.metric(
                "⚠️ No Helmet Probability",
                f"{no_helmet_probability * 100:.1f}%"
            )


        # ====================================================
        # PROGRESS BARS
        # ====================================================

        st.write(
            "🪖 Helmet"
        )

        st.progress(
            helmet_probability
        )


        st.write(
            "⚠️ No Helmet"
        )

        st.progress(
            no_helmet_probability
        )


        # ====================================================
        # EXPLANATION
        # ====================================================

        st.divider()


        if result_type == "safe":

            st.info(
                "The model predicts that a helmet is present."
            )

        else:

            st.warning(
                "The model predicts that a helmet is not present."
            )


# ============================================================
# NO IMAGE
# ============================================================

else:

    st.info(
        "👆 Please upload an image to start helmet detection."
    )

    st.divider()

    st.subheader(
        "How it works"
    )

    step1, step2, step3 = st.columns(3)


    with step1:

        st.write(
            "### 1️⃣ Upload"
        )

        st.write(
            "Upload a JPG, JPEG or PNG image."
        )


    with step2:

        st.write(
            "### 2️⃣ AI Detection"
        )

        st.write(
            "MobileNetV2 analyzes the image."
        )


    with step3:

        st.write(
            "### 3️⃣ Result"
        )

        st.write(
            "The application displays helmet "
            "and no-helmet probabilities."
        )