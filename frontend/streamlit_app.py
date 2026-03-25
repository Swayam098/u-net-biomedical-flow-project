import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import time
import matplotlib.pyplot as plt
import cv2

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="U-Net Biomedical Image Enhancement",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.title("🧠 U-Net Enhancer")

API_URL = st.sidebar.text_input(
    "🔗 Backend API URL",
    value="http://127.0.0.1:5000/predict"
)

SHOW_SVD = st.sidebar.checkbox("Show SVD Baseline", value=True)
SHOW_METRICS = st.sidebar.checkbox("Show Metrics", value=True)

# Theme toggle (Streamlit-native safe)
THEME = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])

# Zoom slider
ZOOM = st.sidebar.slider("🔍 Image Zoom", 0.5, 2.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Domain:** Deep Learning  
    **Sub-domain:** Image Processing  
    **SDG:** Good Health & Well-being
    """
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    f"""
    <h1 style='text-align:center;'>🧠 U-Net–Based Ultrasound Image Enhancement</h1>
    <p style='text-align:center; font-size:18px;'>
    Deep Learning approach for speckle noise suppression and structural preservation
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Ultrasound Image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("Upload an ultrasound image to begin.")
    st.stop()

# --------------------------------------------------
# Load Image
# --------------------------------------------------
image = Image.open(uploaded_file).convert("L")
image_np = np.array(image) / 255.0

# --------------------------------------------------
# Layout
# --------------------------------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Input Image")
    st.image(image_np, use_column_width=False, width=200, clamp=True)

with right:
    RUN = st.button("🚀 Run U-Net Enhancement", use_container_width=True)

# --------------------------------------------------
# Inference with Progress Bar
# --------------------------------------------------
if RUN:
    progress = st.progress(0)
    status = st.empty()

    try:
        status.text("Sending image to backend...")
        progress.progress(20)
        time.sleep(0.2)

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")

        response = requests.post(API_URL, files={"file": img_bytes.getvalue()})
        response.raise_for_status()
        result = response.json()

        progress.progress(70)
        status.text("Processing results...")
        time.sleep(0.2)

        # Store results in session state
        st.session_state.unet_output = np.array(result["unet_output"], dtype=np.float32)
        st.session_state.psnr = float(result.get("psnr", 0.0))
        st.session_state.ssim = float(result.get("ssim", 0.0))
        st.session_state.runtime = float(result.get("runtime", 0.0))
        st.session_state.image_np = image_np

        if SHOW_SVD and "svd_output" in result:
            st.session_state.svd_output = np.array(result["svd_output"], dtype=np.float32)
        else:
            st.session_state.svd_output = None

        progress.progress(100)
        status.text("Done!")

    except Exception as e:
        st.error(f"Backend Error: {e}")
        st.stop()

# Check if inference has been run
if "unet_output" not in st.session_state:
    st.info("Click 'Run U-Net Enhancement' to process the image.")
    st.stop()

# Retrieve from session state
unet_output = st.session_state.unet_output
psnr = st.session_state.psnr
ssim = st.session_state.ssim
runtime = st.session_state.runtime
image_np = st.session_state.image_np
svd_output = st.session_state.svd_output

# --------------------------------------------------
# Medical Grayscale Visualization
# --------------------------------------------------
def show_medical_image(img, title):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)

st.divider()
st.subheader("🖼️ Medical Visualization")

if SHOW_SVD and svd_output is not None:
    c1, c2, c3 = st.columns(3)
    with c1:
        show_medical_image(image_np, "Input Image")
    with c2:
        show_medical_image(unet_output, "U-Net Output")
    with c3:
        show_medical_image(svd_output, "SVD Baseline")
else:
    c1, c2 = st.columns(2)
    with c1:
        show_medical_image(image_np, "Input Image")
    with c2:
        show_medical_image(unet_output, "U-Net Output")

# --------------------------------------------------
# Before / After Slider
# --------------------------------------------------
st.divider()
st.subheader("🔄 Before vs After Comparison")

alpha = st.slider("Slide to compare", 0.0, 1.0, 0.5)

# Resize unet_output back to original image size for blending
original_shape = image_np.shape
# Convert to uint8 for cv2.resize, then back to float for blending
unet_uint8 = (np.clip(unet_output, 0, 1) * 255).astype(np.uint8)
unet_resized_uint8 = cv2.resize(unet_uint8, (original_shape[1], original_shape[0]))
unet_resized = unet_resized_uint8.astype(np.float32) / 255.0

blend = (1 - alpha) * image_np + alpha * unet_resized
show_medical_image(blend, "Before ↔ After")

# --------------------------------------------------
# Metrics
# --------------------------------------------------
if SHOW_METRICS:
    st.divider()
    st.subheader("📊 Quality Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("PSNR (dB)", f"{psnr:.2f}")
    m2.metric("SSIM", f"{ssim:.4f}")
    m3.metric("Runtime (s)", f"{runtime:.2f}")

# --------------------------------------------------
# Download
# --------------------------------------------------
st.divider()
out_img = Image.fromarray((np.clip(unet_output, 0, 1) * 255).astype(np.uint8))
buf = io.BytesIO()
out_img.save(buf, format="PNG")

st.download_button(
    "⬇️ Download Enhanced Image",
    data=buf.getvalue(),
    file_name="unet_enhanced_output.png",
    mime="image/png",
    use_container_width=True
)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.markdown(
    """
    <div style='text-align:center; font-size:14px;'>
    Developed by <b>Swayam Vijay Mehra</b> | CSE (AIML) | SRM Institute of Science and Technology  
    <br>
    Minor Project – Deep Learning & Biomedical Image Processing
    </div>
    """,
    unsafe_allow_html=True
)
