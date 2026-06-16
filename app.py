import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 🎨 PAGE CONFIG (important for professional look)
st.set_page_config(page_title="AI Object Detection", layout="centered")

st.title("🔍 AI Object Detection App")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    # ⭐ Attractive Detect Button
    with col2:
        st.write("")
        st.write("")
        detect = st.button("🚀 Detect Objects")

    if detect:
        with st.spinner("Detecting objects... Please wait ⏳"):
            results = model(np.array(image))

        annotated = results[0].plot()

        st.success("Detection Completed 🎉")

        st.image(annotated, caption="Detected Image", use_container_width=True)

        # ⭐ OBJECT COUNT
        counts = {}

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            counts[class_name] = counts.get(class_name, 0) + 1

        # ⭐ INTERACTIVE UI SECTION
        st.subheader("📊 Object Analysis")

        col1, col2 = st.columns(2)

        items = list(counts.items())

        for i, (obj, count) in enumerate(items):
            if i % 2 == 0:
                col1.metric(label=obj.upper(), value=count)
            else:
                col2.metric(label=obj.upper(), value=count)

        # ⭐ INTERACTIVE TABLE (EXPANDER)
        with st.expander("📋 Detailed Object Table (Click to Expand)"):
            st.table(counts.items())
