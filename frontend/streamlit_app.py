import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="U-Net Biomedical Flow Imaging",
    layout="wide"
)

# -----------------------------
# Title & Description
# -----------------------------
st.title("🧠 U-Net–Based Biomedical Flow Imaging")
st.subheader("Deep Learning–Based Image Processing for Ultrasound Clutter Removal")

st.markdown(
    """
    This application demonstrates a **deep learning–based image processing system**
    that enhances ultrasound flow images using a **U-Net architecture**.
    
    The model removes tissue clutter and noise, improving blood-flow visualization
    without manual parameter tuning.
    
    **Domain:** Deep Learning  
    **Subdomain:** Image Processing (Biomedical)  
    **SDG:** 3 – Good Health and Well-Being
    """
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Controls")
api_url = st.sidebar.text_input(
    "Backend API URL",
    value="http://127.0.0.1:5000/predict"
)

show_svd = st.sidebar.checkbox("Show SVD Baseline (if available)", value=True)
show_metrics = st.sidebar.checkbox("Show Quality Metrics", value=True)

# -----------------------------
# File Upload Section
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Ultrasound Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    st.divider()
    st.subheader("📌 Input Image")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Ultrasound Image", use_column_width=True)

    # -----------------------------
    # Run Inference
    # -----------------------------
    if st.button("🚀 Run AI Filtering"):
        with st.spinner("Running U-Net model..."):
            try:
                # Prepare request
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                files = {"file": img_bytes.getvalue()}

                response = requests.post(api_url, files=files)
                response.raise_for_status()
                result = response.json()

                # Parse results
                output_image = np.array(result["unet_output"])
                svd_image = np.array(result["svd_output"]) if show_svd else None
                psnr = result.get("psnr", None)
                ssim = result.get("ssim", None)
                runtime = result.get("runtime", None)

                # -----------------------------
                # Display Results
                # -----------------------------
                st.divider()
                st.subheader("📊 Results")

                cols = st.columns(3 if show_svd else 2)

                with cols[0]:
                    st.image(image_np, caption="Input Image", use_column_width=True)

                with cols[1]:
                    st.image(output_image, caption="U-Net Filtered Output", use_column_width=True)

                if show_svd:
                    with cols[2]:
                        st.image(svd_image, caption="SVD Baseline Output", use_column_width=True)

                # -----------------------------
                # Metrics
                # -----------------------------
                if show_metrics:
                    st.divider()
                    st.subheader("📈 Quality Metrics")

                    m1, m2, m3 = st.columns(3)
                    if psnr is not None:
                        m1.metric("PSNR", f"{psnr:.2f} dB")
                    if ssim is not None:
                        m2.metric("SSIM", f"{ssim:.4f}")
                    if runtime is not None:
                        m3.metric("Runtime", f"{runtime:.2f} s")

                # -----------------------------
                # Download
                # -----------------------------
                st.divider()
                st.subheader("⬇️ Download Output")

                out_img = Image.fromarray((output_image * 255).astype(np.uint8))
                buf = io.BytesIO()
                out_img.save(buf, format="PNG")

                st.download_button(
                    label="Download U-Net Output Image",
                    data=buf.getvalue(),
                    file_name="unet_filtered_output.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"❌ Error processing image: {e}")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.markdown(
    """
    **Developed by:** Swayam Vijay Mehra  
    **Department:** CSE – AIML  
    **Institution:** SRM Institute of Science and Technology  

    This project demonstrates the integration of **Deep Learning** and
    **Image Processing** for biomedical applications.
    """
)
