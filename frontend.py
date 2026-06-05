import streamlit as st
import requests
import io
from PIL import Image
import base64

# Page Configuration Setup 
st.set_page_config(
    page_title="Kidney Stone Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Kidney Stone Object Detection Dashboard")
st.markdown("---")

# FastAPI endpoint coordinate context definition
API_URL = "http://127.0.0.1:8000/predict"

# 🛠️ Sidebar Configuration Workspace
st.sidebar.header("⚙️ Configuration Hub")
st.sidebar.info("Model Backbone running: **YOLO26m**")

uploaded_file = st.sidebar.file_uploader(
    "Upload Medical Scan Slice (CT/X-ray)", 
    type=["jpg", "jpeg", "png"]
)

# Main Screen Application Split Grid Blueprint
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Read file stream for initial visualization processing rendering 
    bytes_data = uploaded_file.read()
    input_image = Image.open(io.BytesIO(bytes_data))
    
    with col1:
        st.subheader("🖼️ Raw Input Scan")
        st.image(input_image, use_container_width=True, caption="Uploaded Original Scan Matrix")
        
    with col2:
        st.subheader("🎯 YOLO Diagnostic Output")
        with st.spinner("Executing Pipeline Sequence (Preprocessing + Inference)..."):
            try:
                # Prepare payload dictionary stream bytes mapping structure 
                files = {"file": (uploaded_file.name, bytes_data, uploaded_file.type)}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    detections = data.get("detections", [])
                    base64_img_string = data.get("annotated_img", "")
                    
                    if base64_img_string:
                        # Decode back streaming string blocks to functional UI pixel matrices
                        img_bytes = base64.b64decode(base64_img_string)
                        output_image = Image.open(io.BytesIO(img_bytes))
                        
                        st.image(output_image, use_container_width=True, caption="Model Detection Visualization Output")
                        
                        # Render parsed metrics tabular data underneath object visual grids
                        if detections:
                            st.success(f"✅ Found {len(detections)} target stone vector instances!")
                            for idx, obj in enumerate(detections):
                                with st.expander(f"📍 Detection Instance Index #{idx+1}"):
                                    st.write(f"**Target Class Identity (Class ID):** `{obj['class_id']}`")
                                    st.write(f"**Confidence Factor Accuracy Score:** `{obj['score']:.4f}`")
                                    st.write(f"**Bounding Box Pixels Format Array `[X1, Y1, X2, Y2]`:**")
                                    st.code(f"{obj['bbox']}")
                        else:
                            st.warning("⚠️ Diagnostics verified: No target anomalies or stone features located.")
                    else:
                        st.error("Missing prediction array maps inside returned endpoint configurations.")
                else:
                    st.error(f"Backend Server error returned validation code: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Failed to establish communication links with FastAPI server container loop. Please verify `app.py` process is actively live.")
else:
    st.info("💡 Please upload a kidney scan slice using the side controls layout configuration panel to initiate the diagnostic pipeline.")