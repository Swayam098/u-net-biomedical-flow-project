import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import time
import matplotlib.pyplot as plt
import cv2
from preview_effects import PreviewEffects, GaussianBlurBaseline

# Force fresh import of export_reports
import importlib
import sys
if 'export_reports' in sys.modules:
    del sys.modules['export_reports']

from export_reports import ReportExporter, HAS_REPORTLAB

# Debug: Check reportlab status
if not HAS_REPORTLAB:
    print("WARNING: reportlab not available in Streamlit context")
else:
    print("OK: reportlab available in Streamlit context")

# --------------------------------------------------
# Page Config & Styling
# --------------------------------------------------
st.set_page_config(
    page_title="U-Net Biomedical Image Enhancement",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Custom CSS for modern medical design
st.markdown("""
<style>
    :root {
        --primary-blue: #0066CC;
        --light-blue: #E6F0FF;
        --dark-bg: #F8F9FA;
        --border-color: #E0E0E0;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #0066CC !important;
        letter-spacing: -0.5px;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-top: 0.5rem;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        color: #B0B0B0 !important;
    }
    
    /* Remove the white divider/border */
    [data-testid="stMetric"] {
        border: none !important;
        background: none !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F5FF 100%) !important;
        padding: 1.8rem !important;
        border-radius: 14px !important;
        border-left: 5px solid #0066CC !important;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.15) !important;
        border-top: none !important;
        border-right: none !important;
        border-bottom: none !important;
    }
    
    .image-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #E0E0E0;
    }
    
    .header-container {
        background: linear-gradient(135deg, #0066CC 0%, #0052A3 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(0, 102, 204, 0.2);
    }
    
    .header-container h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
    }
    
    .header-container p {
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Remove divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ **Settings & Configuration**")
    
    API_URL = st.text_input(
        "🔗 Backend API URL",
        value="http://127.0.0.1:5000/predict",
        help="Enter the Flask backend API endpoint"
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        SHOW_SVD = st.checkbox("📊 SVD Baseline", value=True)
    with col2:
        SHOW_METRICS = st.checkbox("📈 Metrics", value=True)
    
    st.divider()
    st.markdown("### 📋 **About**")
    st.markdown("""
    **Deep Learning Denoising**  
    • **Model:** U-Net  
    • **Framework:** PyTorch  
    • **Task:** Speckle Suppression  
    • **SDG:** Good Health  
    """)

# --------------------------------------------------
# Header with gradient
# --------------------------------------------------
st.markdown("""
<div class="header-container">
    <h1>🧬 U-Net Ultrasound Enhancement</h1>
    <p>AI-Powered Speckle Noise Suppression with Structural Preservation</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# Upload Section with enhanced styling
# --------------------------------------------------
st.markdown("### 📤 **Upload Ultrasound Image**")

uploaded_file = st.file_uploader(
    "Drag and drop or select PNG / JPG",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.info("👆 Upload a grayscale ultrasound image to begin enhancement", icon="ℹ️")
    st.stop()

# --------------------------------------------------
# Load Image
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image) / 255.0
else:
    st.error("File upload failed")
    st.stop()

# --------------------------------------------------
# Upload Preview & Process Button
# --------------------------------------------------
st.markdown("### 🖼️ **Preview & Process**")

col_preview, col_action = st.columns([2, 1], gap="medium")

with col_preview:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.image(image_np, use_column_width=True, clamp=True)
    st.markdown(f"**Size:** {image_np.shape[1]}×{image_np.shape[0]}px", help="Original image dimensions")
    st.markdown('</div>', unsafe_allow_html=True)

with col_action:
    st.markdown("")
    st.markdown("")
    RUN = st.button(
        "🚀 Process Image",
        use_container_width=True,
        type="primary",
        help="Send image to U-Net model for enhancement"
    )
    st.markdown("---")
    st.caption("⏱️ Processing time: ~0.2-0.3s")

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
    st.info("👆 Click 'Process Image' button to run U-Net enhancement", icon="ℹ️")
    st.stop()

# Retrieve from session state
unet_output = st.session_state.unet_output
psnr = st.session_state.psnr
ssim = st.session_state.ssim
runtime = st.session_state.runtime
image_np = st.session_state.image_np
svd_output = st.session_state.svd_output

# --------------------------------------------------
# Medical Grayscale Visualization Function
# --------------------------------------------------
def show_medical_image(img, title, subtitle=""):
    # Convert to uint8 to avoid memory issues with matplotlib
    if img.dtype != np.uint8:
        img_display = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img_display = img
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
    ax.imshow(img_display, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.5, -0.05, subtitle, ha='center', transform=ax.transAxes, fontsize=10, style='italic')
    ax.axis("off")
    fig.patch.set_facecolor('white')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()
st.markdown("### 📊 **Results & Comparison**")

if SHOW_SVD and svd_output is not None:
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        show_medical_image(image_np, "Original Image", "Raw ultrasound")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        show_medical_image(unet_output, "U-Net Enhanced", "AI-denoised")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        show_medical_image(svd_output, "SVD Baseline", "Reference method")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        show_medical_image(image_np, "Original Image", "Raw ultrasound")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        show_medical_image(unet_output, "U-Net Enhanced", "AI-denoised")
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Interactive Comparison Slider
# --------------------------------------------------
st.divider()
st.markdown("### 🔄 **Interactive Before/After Slider**")
st.markdown("Drag the slider to compare original vs enhanced images")

alpha = st.slider("Enhancement Blend", 0.0, 1.0, 0.5, step=0.05, label_visibility="collapsed")

# Resize unet_output back to original image size for blending
original_shape = image_np.shape
unet_uint8 = (np.clip(unet_output, 0, 1) * 255).astype(np.uint8)
unet_resized_uint8 = cv2.resize(unet_uint8, (original_shape[1], original_shape[0]))
unet_resized = unet_resized_uint8.astype(np.float32) / 255.0

blend = (1 - alpha) * image_np + alpha * unet_resized

# Show blended image
col_blend = st.columns(1)[0]
with col_blend:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    # Convert to uint8 to avoid memory issues
    blend_uint8 = (np.clip(blend, 0, 1) * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax.imshow(blend_uint8, cmap="gray", vmin=0, vmax=255)
    label_text = f"Original" if alpha < 0.1 else f"Enhanced" if alpha > 0.9 else f"Blend ({alpha:.0%})"
    ax.set_title(label_text, fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")
    fig.patch.set_facecolor('white')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Quality Metrics Dashboard
# --------------------------------------------------
if SHOW_METRICS:
    st.divider()
    st.markdown("### 📈 **Quality Metrics & Performance**")
    
    m1, m2, m3 = st.columns(3, gap="medium")
    
    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 PSNR", f"{psnr:.2f} dB", delta="Peak Signal", delta_color="off", help="Peak Signal-to-Noise Ratio - higher is better (>30dB is excellent)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 SSIM", f"{ssim:.4f}", delta="Structural", delta_color="off", help="Structural Similarity Index - closer to 1 is better")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚡ Runtime", f"{runtime:.3f}s", delta="Processing", delta_color="off", help="GPU inference time")
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # Noise Reduction Histogram
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 **Histogram Analysis - Noise Reduction**")
    st.markdown("Comparison of pixel intensity distributions before and after enhancement")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Calculate histograms
    original_hist = np.histogram(image_np.flatten(), bins=256, range=(0, 1))
    enhanced_hist = np.histogram(unet_resized.flatten(), bins=256, range=(0, 1))
    
    bins = np.linspace(0, 1, 257)
    
    # Plot histograms
    ax.hist(image_np.flatten(), bins=256, range=(0, 1), alpha=0.6, label='Original (Noisy)', color='#FF6B6B', edgecolor='none')
    ax.hist(unet_resized.flatten(), bins=256, range=(0, 1), alpha=0.6, label='Enhanced (Denoised)', color='#0066CC', edgecolor='none')
    
    ax.set_xlabel('Pixel Intensity', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Histogram Comparison: Noise Reduction Effect', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    st.pyplot(fig, use_container_width=True)
    
    # Statistics
    col_stat1, col_stat2, col_stat3 = st.columns(3, gap="medium")
    with col_stat1:
        st.markdown(f"""
        <div style='background: #FFE6E6; padding: 1rem; border-radius: 8px; border-left: 4px solid #FF6B6B;'>
            <p style='font-weight: bold; color: #333; margin: 0;'>Original Mean</p>
            <p style='font-size: 1.5rem; color: #FF6B6B; margin: 0.5rem 0 0 0; font-weight: 800;'>{image_np.mean():.3f}</p>
            <p style='font-size: 0.85rem; color: #666; margin: 0;'>Average intensity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown(f"""
        <div style='background: #E6F0FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #0066CC;'>
            <p style='font-weight: bold; color: #333; margin: 0;'>Enhanced Mean</p>
            <p style='font-size: 1.5rem; color: #0066CC; margin: 0.5rem 0 0 0; font-weight: 800;'>{unet_resized.mean():.3f}</p>
            <p style='font-size: 0.85rem; color: #666; margin: 0;'>After denoising</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        reduction = ((image_np.std() - unet_resized.std()) / image_np.std()) * 100 if image_np.std() > 0 else 0
        st.markdown(f"""
        <div style='background: #E6F9F0; padding: 1rem; border-radius: 8px; border-left: 4px solid #00B894;'>
            <p style='font-weight: bold; color: #333; margin: 0;'>Noise Reduction</p>
            <p style='font-size: 1.5rem; color: #00B894; margin: 0.5rem 0 0 0; font-weight: 800;'>{abs(reduction):.1f}%</p>
            <p style='font-size: 0.85rem; color: #666; margin: 0;'>Std dev reduction</p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# A/B Comparison Mode (Optional)
# --------------------------------------------------
st.divider()
st.markdown("### 🔄 **Comparison Mode**")

enable_comparison = st.checkbox("📊 Enable A/B Comparison (U-Net vs SVD vs Gaussian)", value=False)

if enable_comparison and svd_output is not None:
    st.markdown("**Side-by-side comparison of different denoising methods**")
    
    # Generate Gaussian baseline
    gaussian_output = GaussianBlurBaseline.apply(image_np, sigma=2.0)
    gaussian_resized = cv2.resize(
        (np.clip(gaussian_output, 0, 1) * 255).astype(np.uint8),
        (original_shape[1], original_shape[0])
    ).astype(np.float32) / 255.0
    
    # Compute metrics for all methods
    from skimage.metrics import peak_signal_noise_ratio as psnr_calc
    from skimage.metrics import structural_similarity as ssim_calc
    
    try:
        psnr_svd = psnr_calc(image_np, svd_output, data_range=1.0)
        ssim_svd = ssim_calc(image_np, svd_output, data_range=1.0)
        
        psnr_gaussian = psnr_calc(image_np, gaussian_resized, data_range=1.0)
        ssim_gaussian = ssim_calc(image_np, gaussian_resized, data_range=1.0)
    except:
        psnr_svd, ssim_svd = 0, 0
        psnr_gaussian, ssim_gaussian = 0, 0
    
    # 4-panel comparison
    col1, col2, col3, col4 = st.columns(4, gap="small")
    
    with col1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        image_np_uint8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        ax.imshow(image_np_uint8, cmap="gray", vmin=0, vmax=255)
        ax.set_title("Original", fontsize=12, fontweight="bold")
        ax.axis("off")
        fig.patch.set_facecolor('white')
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        unet_resized_uint8 = (np.clip(unet_resized, 0, 1) * 255).astype(np.uint8)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        ax.imshow(unet_resized_uint8, cmap="gray", vmin=0, vmax=255)
        ax.set_title("U-Net Enhanced", fontsize=12, fontweight="bold")
        ax.axis("off")
        fig.patch.set_facecolor('white')
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        if svd_output is not None:
            svd_output_uint8 = (np.clip(svd_output, 0, 1) * 255).astype(np.uint8)
            ax.imshow(svd_output_uint8, cmap="gray", vmin=0, vmax=255)
        ax.set_title("SVD Baseline", fontsize=12, fontweight="bold")
        ax.axis("off")
        fig.patch.set_facecolor('white')
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        gaussian_resized_uint8 = (np.clip(gaussian_resized, 0, 1) * 255).astype(np.uint8)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        ax.imshow(gaussian_resized_uint8, cmap="gray", vmin=0, vmax=255)
        ax.set_title("Gaussian Blur", fontsize=12, fontweight="bold")
        ax.axis("off")
        fig.patch.set_facecolor('white')
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics comparison table
    st.markdown("**Metrics Comparison**")
    comparison_data = {
        "Method": ["U-Net", "SVD", "Gaussian"],
        "PSNR (dB)": [f"{psnr:.2f}", f"{psnr_svd:.2f}", f"{psnr_gaussian:.2f}"],
        "SSIM": [f"{ssim:.4f}", f"{ssim_svd:.4f}", f"{ssim_gaussian:.4f}"]
    }
    st.dataframe(comparison_data, use_container_width=True)
elif enable_comparison:
    st.warning("⚠️ SVD output not available - enable SVD in sidebar")

# --------------------------------------------------
# Real-Time Preview Controls
# --------------------------------------------------
st.divider()
st.markdown("### 🎨 **Real-Time Preview Adjustments**")

st.markdown("**Adjust the enhanced image on-the-fly**")

col_intensity, col_blur, col_contrast = st.columns(3, gap="medium")

with col_intensity:
    intensity = st.slider("🔆 Intensity", 0.8, 1.2, 1.0, 0.05, help="Brightness adjustment")

with col_blur:
    blur = st.slider("📊 Blur", 0.0, 5.0, 0.0, 0.5, help="Gaussian blur amount")

with col_contrast:
    contrast = st.slider("🎨 Contrast", 0.8, 1.5, 1.0, 0.05, help="Contrast enhancement")

# Apply preview effects
preview_config = PreviewEffects.create_preview_config(
    intensity=intensity,
    blur=blur,
    contrast=contrast,
    saturation=0.0
)

preview_image = PreviewEffects.apply_all_effects(
    unet_resized,
    **preview_config
)

# Display preview
st.markdown("**Live Preview**")
st.markdown('<div class="image-card">', unsafe_allow_html=True)
preview_image_uint8 = (np.clip(preview_image, 0, 1) * 255).astype(np.uint8)
fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
ax.imshow(preview_image_uint8, cmap="gray", vmin=0, vmax=255)
ax.set_title("Preview with Adjustments", fontsize=14, fontweight="bold")
ax.axis("off")
fig.patch.set_facecolor('white')
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Export Reports
# --------------------------------------------------
st.divider()
st.markdown("### 📄 **Export Reports**")

st.markdown("**Generate professional reports in PDF or PNG format**")

col_png, col_pdf = st.columns(2, gap="medium")

exporter = ReportExporter()

with col_png:
    try:
        png_data = exporter.export_png(image_np, unet_resized, psnr, ssim, runtime)
        st.download_button(
            "🖼️ Export as PNG",
            data=png_data,
            file_name="unet_report.png",
            mime="image/png",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"PNG export error: {e}")

with col_pdf:
    try:
        pdf_data = exporter.export_pdf(image_np, unet_resized, psnr, ssim, runtime)
        if pdf_data:
            st.download_button(
                "📋 Export as PDF",
                data=pdf_data,
                file_name="unet_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("⚠️ PDF export not available (reportlab not installed)")
    except Exception as e:
        st.error(f"PDF export error: {str(e)}")
        with st.expander("Debug info"):
            st.code(str(e))

# Download enhanced image
st.markdown("---")
st.markdown("### 💾 **Download Enhanced Image**")

out_img = Image.fromarray((np.clip(unet_output, 0, 1) * 255).astype(np.uint8))
buf = io.BytesIO()
out_img.save(buf, format="PNG")

col_download = st.columns(1)[0]
with col_download:
    st.download_button(
        "⬇️ Download Enhanced Image",
        data=buf.getvalue(),
        file_name="unet_enhanced_output.png",
        mime="image/png",
        use_container_width=True
    )

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0; font-size: 0.9rem;'>
    <p>🧬 <b>U-Net Biomedical Image Enhancement</b> | AI-Powered Medical Imaging</p>
    <p>Powered by <b>PyTorch</b> • Designed for speckle noise suppression in ultrasound</p>
    <p style='font-size: 0.85rem; color: #999;'>© 2026 | Academic Project</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.markdown(
    """
    <div style='text-align:center; font-size:14px;'>
    Developed by <b>Swayam Vijay Mehra AND Prateek Shulka</b> | CSE (AIML) | SRM Institute of Science and Technology  
    <br>
    Minor Project – Deep Learning & Biomedical Image Processing
    </div>
    """,
    unsafe_allow_html=True
)
