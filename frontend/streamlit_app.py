import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

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

    The model removes tissue clutter and noise, improving visualization of
    flow-related structures without manual parameter tuning.

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

show_svd = st.sidebar.checkbox("Show SVD Baseline", value=True)
show_metrics = st.sidebar.checkbox("Show Quality Metrics", value=True)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Ultrasound Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    st.divider()
    st.subheader("📌 Input Image")

    st.image(image, caption="Original Ultrasound Image", width=350)

    # -----------------------------
    # Run Inference
    # -----------------------------
    if st.button("🚀 Run AI Filtering"):
        with st.spinner("Running U-Net model..."):
            try:
                # Prepare request
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")

                response = requests.post(
                    api_url,
                    files={"file": img_bytes.getvalue()},
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()

                # Parse results
                unet_output = np.array(result["unet_output"])
                svd_output = np.array(result["svd_output"])

                psnr = result.get("psnr")
                ssim = result.get("ssim")
                runtime = result.get("runtime")

                # -----------------------------
                # Display normalization
                # -----------------------------
                def normalize(img):
                    img = img.astype(np.float32)
                    return (img - img.min()) / (img.max() - img.min() + 1e-8)

                input_display = normalize(image_np)
                unet_display = normalize(unet_output)
                svd_display = normalize(svd_output)

                # -----------------------------
                # Results Display (FIXED)
                # -----------------------------
                st.divider()
                st.subheader("📊 Results Comparison")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        input_display,
                        caption="Input Image",
                        width=300
                    )

                with col2:
                    st.image(
                        unet_display,
                        caption="U-Net Filtered Output",
                        width=300
                    )

                with col3:
                    if show_svd:
                        st.image(
                            svd_display,
                            caption="SVD Baseline Output",
                            width=300
                        )
                    else:
                        st.empty()

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
                # Download Output
                # -----------------------------
                st.divider()
                st.subheader("⬇️ Download U-Net Output")

                out_img = Image.fromarray((unet_display * 255).astype(np.uint8))
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

    This project demonstrates the integration of **Deep Learning**
    and **Image Processing** for biomedical imaging applications.
    """
)
