import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Object Detection")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image)

    if st.button("Detect"):
        results = model(np.array(image))

        annotated = results[0].plot()

        st.image(annotated, channels="BGR") 
