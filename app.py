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

    st.image(image, caption="Original Image")

    if st.button("Detect"):
        results = model(np.array(image))

        annotated = results[0].plot()
        st.image(annotated, caption="Detected Image", channels="BGR")

        # ⭐ OBJECT COUNT ADDED HERE
        counts = {}

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            counts[class_name] = counts.get(class_name, 0) + 1

        st.subheader("Object Count")
        st.table(counts.items())
