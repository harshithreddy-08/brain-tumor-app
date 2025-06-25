import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import time
import base64

# Load model
model = load_model("vgg_model.h5")

# App Config
st.set_page_config(page_title="MEDScan.AI", layout="centered")

# Background styling
st.markdown(
    """
    <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1588776814546-ec7e077bf0f3');
            background-size: cover;
            background-attachment: fixed;
            color: white;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True
)

# Header
st.title("🧠 MEDScan.AI - Brain Tumor Detector")
st.markdown("Upload your MRI scan and enter patient details for a diagnostic report.")
st.markdown("---")

# Form
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("👤 Patient Name")
        age = st.number_input("🎂 Age", min_value=1, max_value=120)
        gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"])
    with col2:
        contact = st.text_input("📞 Contact Number")
        scan_date = st.date_input("📅 Scan Date")

    uploaded_file = st.file_uploader("📷 Upload Brain MRI Image", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("🧪 Diagnose Now")

# Diagnose
if submit:
    if uploaded_file is not None and name:
        st.markdown("### 🔍 Diagnosing...")
        image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

        prediction = model.predict(img_array)
        class_names = ["No Tumor", "Tumor"]
        result = class_names[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)

        time.sleep(2)

        st.success("✅ Diagnosis Completed")
        st.markdown("### 🧾 Diagnostic Report")
        st.image(image, caption="MRI Scan", use_column_width=True)

        # Report HTML
        report_html = f"""
        <html>
        <head><style>
            body {{
                font-family: Arial;
                margin: 20px;
                padding: 10px;
                background-color: #f8f9fa;
                color: #212529;
            }}
            h2 {{
                color: #2c3e50;
            }}
            .highlight {{
                font-weight: bold;
                color: #2c3e50;
            }}
        </style></head>
        <body>
            <h2>MEDScan.AI - Patient Diagnostic Report</h2>
            <p><span class='highlight'>Name:</span> {name}</p>
            <p><span class='highlight'>Age:</span> {age}</p>
            <p><span class='highlight'>Gender:</span> {gender}</p>
            <p><span class='highlight'>Contact:</span> {contact}</p>
            <p><span class='highlight'>Scan Date:</span> {scan_date.strftime('%B %d, %Y')}</p>
            <p><span class='highlight'>Diagnosis:</span> {result}</p>
            <p><span class='highlight'>Confidence:</span> {confidence}%</p>
            <p><span class='highlight'>Doctor Notes:</span> {"Immediate consultation recommended." if result == "Tumor" else "No signs of brain tumor detected."}</p>
        </body>
        </html>
        """

        # Display in Streamlit
        st.markdown(report_html, unsafe_allow_html=True)
        st.balloons()

        # Downloadable report
        b64 = base64.b64encode(report_html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="medscan_report.html">📥 Download Diagnostic Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("⚠️ Please fill all patient details and upload an image.")

