# dashboard/app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

# --- Utils ---

def preprocess_image(img):
    img_resized = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_trans = np.transpose(img_norm, (2, 0, 1))  # CHW
    return np.expand_dims(img_trans, axis=0)

def load_onnx_model(path="training/exports/ct_defect.onnx"):
    return ort.InferenceSession(path)

def run_onnx_inference(session, img_tensor):
    inputs = {session.get_inputs()[0].name: img_tensor}
    outputs = session.run(None, inputs)
    return outputs[0]  # shape: (batch, num_detections, 6)

def draw_predictions(img, preds, conf_thresh=0.4):
    for pred in preds:
        for *xyxy, conf, cls in pred:
            if conf > conf_thresh:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Defect: {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

# --- Streamlit UI ---

st.set_page_config(page_title="Industrial CT Defect Detector", layout="centered")
st.title("🧠 Industrial CT Defect Detection")
st.markdown("Upload a CT image and run ONNX-powered AI inference to detect structural defects.")

uploaded_file = st.file_uploader("📤 Upload CT image (.png, .jpg)", type=["png", "jpg"])
conf_threshold = st.slider("Confidence threshold", 0.2, 0.9, 0.4, step=0.05)

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Run Defect Detection (ONNX)"):
        st.info("Running ONNX inference...")
        session = load_onnx_model()
        input_tensor = preprocess_image(image)
        output = run_onnx_inference(session, input_tensor)

        boxes = output[0][:, :4] * 256  # de-normalize to image size
        confs = output[0][:, 4]
        classes = output[0][:, 5]
        preds = [[*boxes[i], confs[i], classes[i]] for i in range(len(confs))]

        result_img = draw_predictions(image.copy(), [preds], conf_thresh=conf_threshold)
        st.image(result_img, caption="Detected Defects", use_column_width=True)

