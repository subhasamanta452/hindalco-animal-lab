"""Local Streamlit UI for the saved animal classifier."""

import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_utils import load_model, predict_image  # noqa: E402

st.set_page_config(page_title="Animal Predictor", page_icon="🐾")
st.title("Animal Predictor")
st.write("Upload a photo and the local MobileNetV2 model will predict the animal.")


@st.cache_resource
def get_model():
    return load_model(PROJECT_ROOT / "models")


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Predict", type="primary"):
        try:
            model, classes = get_model()
            result = predict_image(model, classes, image)
            st.success(f"Prediction: {result['animal']}")
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.subheader("Scores by animal")
            st.bar_chart(result["all_scores"], horizontal=True)
            if result["confidence"] < 0.60:
                st.warning("The model is less than 60% confident in this prediction.")
        except Exception as error:
            st.error(f"Could not make a prediction: {error}")
else:
    st.info("Upload an image to begin.")