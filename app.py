"""Streamlit app for the Spam Message Detector.

Run with:
    streamlit run app.py
(Train the model first with: python train_model.py)
"""
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path("model") / "spam_model.joblib"


@st.cache_resource
def load_model():
    """Load the saved pipeline once and reuse it across interactions."""
    return joblib.load(MODEL_PATH)


st.title("Spam Message Detector")

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Please run `python train_model.py` first, "
        "then restart the app."
    )
    st.stop()

model = load_model()

message = st.text_area("Enter a message to check:", height=150)

if st.button("Detect Spam"):
    if not message.strip():
        st.warning("Please enter a message before clicking 'Detect Spam'.")
    else:
        prediction = model.predict([message])[0]
        if prediction == "spam":
            st.error("🚨 SPAM")
        else:
            st.success("✅ NOT SPAM")

st.caption(
    "Note: Predictions can be incorrect. This is a simple demo model "
    "and should not be relied on for important decisions."
)
