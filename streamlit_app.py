
import streamlit as st
import pydicom
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import os
import json
import pandas as pd
from tensorflow.keras.models import load_model

def make_pseudo_color(img_2d):
    img_uint8 = (img_2d * 255).astype(np.uint8)

    ch1 = img_uint8

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch2 = clahe.apply(img_uint8)

    p1, p99 = np.percentile(img_uint8, (1, 99))
    ch3 = np.clip(img_uint8, p1, p99)
    ch3 = ((ch3 - p1) / (p99 - p1 + 1e-8) * 255).astype(np.uint8)

    pseudo_img = np.stack([ch1, ch2, ch3], axis=-1).astype(np.float32)

    return pseudo_img


def preprocess_dicom(uploaded_file):

    ds = pydicom.dcmread(uploaded_file, force=True)

    # =========================
    # Original DICOM
    # =========================
    raw_img = ds.pixel_array.astype(np.float32)

    img = raw_img.copy()

    # =========================
    # HU Conversion
    # =========================
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    # =========================
    # Windowing
    # =========================
    WL, WW = 40, 350

    low = WL - WW / 2
    high = WL + WW / 2

    img = np.clip(img, low, high)

    # =========================
    # Normalization
    # =========================
    img = (img - low) / (high - low + 1e-8)

    windowed_img = img.copy()

    # =========================
    # Resize
    # =========================
    img = cv2.resize(
        img,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # =========================
    # Pseudo-color
    # =========================
    pseudo_img = make_pseudo_color(img)

    # =========================
    # ResNet preprocessing
    # =========================
    model_img = preprocess_input(
        pseudo_img.copy()
    )

    return (
        raw_img,
        windowed_img,
        pseudo_img,
        model_img
    )



st.set_page_config(
    page_title="CT-Based Intelligent System",
    page_icon="🧠",
    layout="wide"
)

MODEL_DIR = "/content/drive/MyDrive/neck_ct_saved_model"
MODELS_DIR = os.path.join(MODEL_DIR, "models")

@st.cache_resource
def load_ensemble_models():
    seeds = [1, 7, 21, 42, 100]
    models = {}

    for seed in seeds:
        model_path = os.path.join(MODELS_DIR, f"model_seed_{seed}.keras")
        models[seed] = load_model(model_path, compile=False)

    return models


@st.cache_data
def load_config_files():
    with open(os.path.join(MODEL_DIR, "model_config.json"), "r") as f:
        config = json.load(f)

    with open(os.path.join(MODEL_DIR, "flip_flags.json"), "r") as f:
        flip_flags = json.load(f)

    return config, flip_flags


def predict_slice_ensemble(model_img, models, flip_flags):
    x = np.expand_dims(model_img, axis=0)
    probs = []

    for seed, model in models.items():
        prob = float(model.predict(x, verbose=0).ravel()[0])

        if str(seed) in flip_flags and flip_flags[str(seed)]:
            prob = 1 - prob

        probs.append(prob)

    final_prob = float(np.mean(probs))
    return final_prob, probs


# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #F7FAFC;
}
.title-box {
    background: linear-gradient(90deg, #00796B, #0097A7);
    padding: 25px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}
.card {
    background-color: white;
    color: #1F2937;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 18px;
    font-size: 18px;
}
.card p {
    color: #1F2937;
}
.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
.normal-box {
    background-color: #E8F5E9;
    color: #2E7D32;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
.abnormal-box {
    background-color: #FFEBEE;
    color: #C62828;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
.small-text {
    color: #607D8B;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("""
<div class="title-box">
    <h1>CT-Based Intelligent System</h1>
    <h3>Head & Neck CT Abnormality Classification</h3>
    <p>AI-assisted patient-level analysis using ResNet50 ensemble voting</p>
</div>
""", unsafe_allow_html=True)

models = load_ensemble_models()
config, flip_flags = load_config_files()

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Decision Settings")

slice_threshold = st.sidebar.slider(
    "Slice Threshold",
    0.0,
    1.0,
    float(config["slice_threshold"]),
    0.01
)

patient_threshold = st.sidebar.slider(
    "Patient Threshold",
    0.0,
    1.0,
    float(config["patient_threshold"]),
    0.01
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write("Model: ResNet50 Ensemble")
st.sidebar.write("Number of models: 5")
st.sidebar.write("Input size: 224 × 224 × 3")
st.sidebar.write("Mode: Patient-level decision")

# =========================
# Upload Section
# =========================
st.markdown("## 📂 Upload CT Study")

upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    zip_file = st.file_uploader(
        "Upload ZIP file containing DICOM slices",
        type=["zip"]
    )

with upload_col2:
    dicom_files = st.file_uploader(
        "Or upload DICOM files directly",
        type=["dcm", "dicom"],
        accept_multiple_files=True
    )

# =========================
# Collect uploaded DICOMs
# =========================
uploaded_dicom_list = []

if dicom_files:
    uploaded_dicom_list = dicom_files

elif zip_file:
    import zipfile
    import tempfile

    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                ds = pydicom.dcmread(file_path, force=True)

                if hasattr(ds, "PixelData"):
                    uploaded_dicom_list.append(file_path)

            except:
                pass

# =========================
# Preview first slice
# =========================
if uploaded_dicom_list:
  raw_img, windowed_img, pseudo_img, model_img = preprocess_dicom(
        uploaded_dicom_list[0]
    )

# =========================
# Patient Information
# =========================
st.markdown("## 🧾 Patient Information")

patient_id = "HN-001"
study_date = "--"
modality = "CT"

if uploaded_dicom_list:
    try:
        ds_info = pydicom.dcmread(uploaded_dicom_list[0], force=True, stop_before_pixels=True)
        patient_id = str(getattr(ds_info, "PatientID", "HN-001"))
        study_date = str(getattr(ds_info, "StudyDate", "--"))
        modality = str(getattr(ds_info, "Modality", "CT"))
    except:
        pass

processed_slices_display = len(uploaded_dicom_list) if uploaded_dicom_list else "--"

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.metric("Patient ID", patient_id)

with p2:
    st.metric("Modality", modality)

with p3:
    st.metric("Study Date", study_date)

with p4:
    st.metric("Processed Slices", processed_slices_display)

# =========================
# Preview Section
# =========================
st.markdown("## 🖼 CT Processing Preview")

if uploaded_dicom_list:
    c1, c2, c3 = st.columns(3)

    with c1:
        raw_preview = raw_img.astype(np.float32)
        raw_preview = (raw_preview - raw_preview.min()) / (
            raw_preview.max() - raw_preview.min() + 1e-8
        )

        st.image(
            raw_preview,
            caption="Original CT",
            use_container_width=True,
            clamp=True
        )

    with c2:
        st.image(
            windowed_img,
            caption="Windowed CT",
            use_container_width=True,
            clamp=True
        )

    with c3:
        st.image(
            pseudo_img.astype(np.uint8),
            caption="Pseudo-color Input",
            use_container_width=True
        )
else:
    st.info("Upload DICOM files or a ZIP file to preview the CT preprocessing steps.")

# =========================
# Analysis Button
# =========================
st.markdown("## 🧠 AI Analysis")

if st.button("Run Patient-Level Analysis"):

    if not uploaded_dicom_list:

        st.warning("Please upload DICOM files or a ZIP file first.")

    else:

        with st.spinner("Processing CT slices and running AI ensemble..."):

            progress = st.progress(0)

            slice_results = []

            total_files = len(uploaded_dicom_list)

            for i, dicom_item in enumerate(uploaded_dicom_list):

                raw_i, windowed_i, pseudo_i, model_i = preprocess_dicom(
                    dicom_item
                )

                final_prob, seed_probs = predict_slice_ensemble(
                    model_i,
                    models,
                    flip_flags
                )

                vote = 1 if final_prob >= slice_threshold else 0

                slice_results.append({
                    "Slice": i + 1,
                    "P(Abnormal)": round(final_prob, 4),
                    "Vote": "Positive" if vote == 1 else "Negative"
                })

                progress.progress((i + 1) / total_files)

        # =========================
        # Patient-level stats
        # =========================

        total_slices = len(slice_results)

        positive_slices = sum(
            1 for r in slice_results
            if r["Vote"] == "Positive"
        )

        positive_ratio = (
            positive_slices / total_slices
            if total_slices > 0 else 0
        )

        final_result = (
            "Abnormal"
            if positive_ratio >= patient_threshold
            else "Normal"
        )

        # =========================
        # Slice predictions
        # =========================

        st.markdown("## 🧩 Slice-Level Predictions")

        df_results = pd.DataFrame(slice_results)

        st.dataframe(
            df_results,
            use_container_width=True
        )

        # =========================
        # Summary
        # =========================

        st.markdown("## 📊 Patient-Level Summary")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Total Slices", total_slices)

        with m2:
            st.metric("Positive Slices", positive_slices)

        with m3:
            st.metric(
                "Positive Slice Ratio",
                f"{positive_ratio:.2f}"
            )

        # =========================
        # Final classification
        # =========================

        st.markdown("## ✅ Final Classification")

        if final_result == "Abnormal":

            st.markdown("""
            <div class="abnormal-box">
                🔴 FINAL CLASSIFICATION: ABNORMAL
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="normal-box">
                🟢 FINAL CLASSIFICATION: NORMAL
            </div>
            """, unsafe_allow_html=True)

        # =========================
        # Decision explanation
        # =========================

        st.markdown("## 💡 Decision Explanation")

        st.info(
            f"{positive_slices} out of {total_slices} slices exceeded the slice threshold. "
            f"The positive slice ratio is {positive_ratio:.2f}. "
            f"Since this value is {'greater than or equal to' if positive_ratio >= patient_threshold else 'less than'} "
            f"the patient threshold ({patient_threshold:.2f}), "
            f"the patient is classified as {final_result}."
        )

        st.markdown("---")

        st.caption(
            "For research and educational purposes only. "
            "Not intended for standalone clinical diagnosis."
        )
