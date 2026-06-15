import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2

# -----------------------------
# Load YOLOv5 model (FREE pretrained)
# -----------------------------
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

# -----------------------------
# Detection function
# -----------------------------
def detect(image):
    img = np.array(image)

    results = model(img)

    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        conf = row['confidence']

        if conf > 0.4:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    return img

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔍 AI Object Detection App (PyTorch YOLOv5)")
st.write("Upload image and detect objects using FREE YOLOv5 model")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Detect Objects"):
        result = detect(image)
        st.image(result, caption="Detected Objects", use_container_width=True)
