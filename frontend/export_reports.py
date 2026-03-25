"""
Export module for generating reports in PDF and PNG formats
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from datetime import datetime
from typing import Tuple, Optional
import cv2

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib import colors
    HAS_REPORTLAB = True
    print("✅ [export_reports] reportlab loaded successfully")
except ImportError as e:
    HAS_REPORTLAB = False
    print(f"⚠️ [export_reports] reportlab import failed: {e}")


class ReportExporter:
    """Export analysis reports in various formats"""
    
    @staticmethod
    def _normalize_image(img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 uint8"""
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)
    
    @staticmethod
    def _create_comparison_image(
        original: np.ndarray,
        enhanced: np.ndarray,
        size: Tuple[int, int] = (400, 300)
    ) -> Image.Image:
        """Create side-by-side comparison image"""
        # Normalize
        orig_img = ReportExporter._normalize_image(original)
        enh_img = ReportExporter._normalize_image(enhanced)
        
        # Resize
        orig_resized = cv2.resize(orig_img, size)
        enh_resized = cv2.resize(enh_img, size)
        
        # Convert to RGB for PIL
        orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2RGB)
        enh_rgb = cv2.cvtColor(enh_resized, cv2.COLOR_GRAY2RGB)
        
        # Create side-by-side
        comparison = np.hstack([orig_rgb, enh_rgb])
        
        # Convert to PIL
        return Image.fromarray(comparison)
    
    @staticmethod
    def _create_histogram_image(
        original: np.ndarray,
        enhanced: np.ndarray,
        size: Tuple[int, int] = (600, 300)
    ) -> Image.Image:
        """Create histogram comparison image"""
        fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
        
        ax.hist(original.flatten(), bins=256, range=(0, 1), alpha=0.6, label='Original', color='#FF6B6B', edgecolor='none')
        ax.hist(enhanced.flatten(), bins=256, range=(0, 1), alpha=0.6, label='Enhanced', color='#0066CC', edgecolor='none')
        
        ax.set_xlabel('Pixel Intensity', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title('Histogram Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        fig.patch.set_facecolor('white')
        
        # Convert to PIL using buffer_rgba (newer matplotlib API)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        
        # Use buffer_rgba() for compatibility with newer matplotlib
        rgba_buffer = renderer.buffer_rgba()
        size_tuple = canvas.get_width_height()
        hist_img = Image.frombytes('RGBA', size_tuple, rgba_buffer)
        # Convert RGBA to RGB
        hist_img = hist_img.convert('RGB')
        plt.close(fig)
        
        return hist_img
    
    @staticmethod
    def export_png(
        original: np.ndarray,
        enhanced: np.ndarray,
        psnr: float,
        ssim: float,
        runtime: float,
        filename: str = "report.png"
    ) -> bytes:
        """Export as PNG composite image"""
        
        # Create canvas (1200x1600)
        canvas_width, canvas_height = 1200, 1600
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
        draw = ImageDraw.Draw(canvas)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 32)
            header_font = ImageFont.truetype("arial.ttf", 24)
            text_font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            header_font = title_font
            text_font = title_font
            small_font = title_font
        
        y_offset = 30
        
        # Title
        draw.text((canvas_width//2, y_offset), "U-Net Enhancement Report", 
                 fill='black', font=title_font, anchor="mm")
        y_offset += 60
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((30, y_offset), f"Generated: {timestamp}", fill='#666666', font=small_font)
        y_offset += 40
        
        # Comparison images
        comparison = ReportExporter._create_comparison_image(original, enhanced)
        comparison = comparison.resize((1140, 300))
        canvas.paste(comparison, (30, y_offset))
        y_offset += 340
        
        # Labels
        draw.text((30, y_offset), "Original", fill='#666666', font=text_font)
        draw.text((600, y_offset), "Enhanced", fill='#666666', font=text_font)
        y_offset += 40
        
        # Metrics
        draw.text((30, y_offset), "Quality Metrics:", fill='black', font=header_font, weight='bold')
        y_offset += 50
        
        metrics = [
            f"PSNR (dB): {psnr:.2f}",
            f"SSIM: {ssim:.4f}",
            f"Runtime (s): {runtime:.3f}",
        ]
        
        for metric in metrics:
            draw.text((60, y_offset), metric, fill='#0066CC', font=text_font)
            y_offset += 35
        
        # Histogram
        y_offset += 20
        histogram = ReportExporter._create_histogram_image(original, enhanced)
        histogram = histogram.resize((1140, 300))
        canvas.paste(histogram, (30, y_offset))
        y_offset += 340
        
        # Footer
        footer_text = "U-Net Biomedical Image Enhancement | AI-Powered Denoising"
        draw.text((canvas_width//2, canvas_height - 30), footer_text, 
                 fill='#999999', font=small_font, anchor="mm")
        
        # Save to bytes
        buf = io.BytesIO()
        canvas.save(buf, format='PNG')
        return buf.getvalue()
    
    @staticmethod
    def export_pdf(
        original: np.ndarray,
        enhanced: np.ndarray,
        psnr: float,
        ssim: float,
        runtime: float,
        filename: str = "report.pdf"
    ) -> Optional[bytes]:
        """Export as PDF report"""
        
        if not HAS_REPORTLAB:
            return None
        
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=0.5*inch, leftMargin=0.5*inch,
                               topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#0066CC'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#0066CC'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        )
        
        # Title
        elements.append(Paragraph("U-Net Biomedical Image Enhancement", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated: {timestamp}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Images section
        elements.append(Paragraph("Enhancement Results", heading_style))
        
        # Save comparison image
        comparison = ReportExporter._create_comparison_image(original, enhanced, size=(250, 200))
        comparison_buf = io.BytesIO()
        comparison.save(comparison_buf, format='PNG')
        comparison_buf.seek(0)
        
        rl_image = RLImage(comparison_buf, width=4*inch, height=1.2*inch)
        elements.append(rl_image)
        elements.append(Spacer(1, 0.1*inch))
        
        # Metrics table
        elements.append(Paragraph("Quality Metrics", heading_style))
        
        metrics_data = [
            ['Metric', 'Value'],
            ['PSNR (dB)', f"{psnr:.2f}"],
            ['SSIM', f"{ssim:.4f}"],
            ['Runtime (s)', f"{runtime:.3f}"],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Histogram
        elements.append(Paragraph("Histogram Analysis", heading_style))
        
        histogram = ReportExporter._create_histogram_image(original, enhanced)
        histogram_buf = io.BytesIO()
        histogram.save(histogram_buf, format='PNG')
        histogram_buf.seek(0)
        
        rl_histogram = RLImage(histogram_buf, width=5*inch, height=2*inch)
        elements.append(rl_histogram)
        
        # Build PDF
        try:
            doc.build(elements)
            return buf.getvalue()
        except Exception as e:
            print(f"PDF generation error: {e}")
            return None


def test_exports():
    """Test export functions"""
    print("Testing export functions...")
    
    # Create dummy images
    original = np.random.rand(256, 256)
    enhanced = np.random.rand(256, 256)
    
    exporter = ReportExporter()
    
    # Test PNG
    try:
        png_data = exporter.export_png(original, enhanced, 39.84, 0.9916, 0.25)
        print(f"✓ PNG export: {len(png_data)} bytes")
    except Exception as e:
        print(f"✗ PNG export failed: {e}")
    
    # Test PDF
    try:
        if HAS_REPORTLAB:
            pdf_data = exporter.export_pdf(original, enhanced, 39.84, 0.9916, 0.25)
            if pdf_data:
                print(f"✓ PDF export: {len(pdf_data)} bytes")
            else:
                print("✗ PDF export returned None")
        else:
            print("⚠ reportlab not installed")
    except Exception as e:
        print(f"✗ PDF export failed: {e}")


if __name__ == "__main__":
    test_exports()
