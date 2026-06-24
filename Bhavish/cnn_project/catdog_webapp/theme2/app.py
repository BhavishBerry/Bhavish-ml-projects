"""
Cat vs Dog Classifier - Theme 2 (Sunset Pop)

PLUG AND PLAY:
1. Change NAME below to your own name.
2. A trained model.h5 is already included in this folder - works out of the box.
   (Swap it for your own model.h5 if you trained your own.)
3. Run:  streamlit run app.py
That's it - no GitHub, no deployment, runs fully on your own laptop.
"""

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- EDIT THIS ----------------
NAME = "Your Name"
# --------------------------------------------

PAGE_TITLE = f"{NAME} | Cat vs Dog Classifier"
MODEL_PATH = "model.h5"
IMG_SIZE = (64, 64)

st.set_page_config(page_title=PAGE_TITLE, page_icon="🐾", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #ffd6a5 0%, #ff6b6b 60%, #c44569 100%);
        color: #2d1b1b;
    }
    h1, h2, h3, p, label, .stMarkdown { color: #2d1b1b !important; }
    .stButton>button {
        background-color: #ffffff; color: #c44569; border: none;
        border-radius: 8px; font-weight: 700;
    }
    .result-box {
        background-color: rgba(255,255,255,0.85); border: 2px solid #c44569;
        border-radius: 16px; padding: 1.2rem; text-align: center;
        margin-top: 1rem;
    }
    .result-box h2 { color: #c44569 !important; }
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255,255,255,0.6); border: 1px dashed #c44569;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


st.title("🐾 Cat vs Dog Classifier")
st.caption(f"Built by {NAME} — a CNN trained from scratch on real cat & dog photos")

uploaded = st.file_uploader("Upload a photo of a cat or dog", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your upload", use_container_width=True)

    try:
        model = get_model()
    except Exception:
        st.error(f"Couldn't find '{MODEL_PATH}'. Put your trained model file in this folder as 'model.h5'.")
        st.stop()

    resized = img.resize(IMG_SIZE)
    arr = np.expand_dims(np.array(resized) / 255.0, axis=0)
    pred = model.predict(arr)[0][0]

    label = "Dog 🐶" if pred >= 0.5 else "Cat 🐱"
    confidence = pred if pred >= 0.5 else 1 - pred

    st.markdown(
        f"""
        <div class="result-box">
            <h2>{label}</h2>
            <p>Confidence: {confidence * 100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Upload an image above to get a prediction.")
