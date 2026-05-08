# Ultrasound Image Enhancement Using U-Net with Uncertainty Estimation

## Abstract
Ultrasound images are often degraded by speckle noise, which reduces lesion visibility and makes downstream interpretation less reliable. This paper presents a U-Net-based denoising framework for ultrasound image enhancement together with Monte Carlo dropout for predictive uncertainty estimation. The proposed system is evaluated on BUSI-style ultrasound imagery using PSNR, SSIM, and MAE as quality measures. The implementation combines a PyTorch backend for model inference with a Streamlit frontend for interactive visualization, enabling both denoised outputs and uncertainty maps to be inspected in real time. Experimental results from the current project artifacts show that the enhanced U-Net configuration improves image quality over the baseline MSE-trained model, with PSNR increasing from 20.04 dB to 27.02 dB and SSIM increasing from 0.488 to 0.742 on the recorded comparison set. In addition to higher reconstruction fidelity, the uncertainty maps provide a reliability cue for regions where the model prediction is less confident. These results indicate that U-Net-based denoising with uncertainty estimation is a practical approach for robust ultrasound image enhancement and deployment.

## Keywords
Ultrasound Imaging, Image Denoising, U-Net, Speckle Noise, Uncertainty Estimation, Monte Carlo Dropout

## I. Introduction
Ultrasound imaging is widely used in clinical practice because it is non-invasive, relatively low-cost, and safe for repeated examination. It plays an important role in the assessment of soft tissue structures, breast lesions, and many other diagnostic tasks. However, ultrasound images are frequently affected by speckle noise, a multiplicative granular artifact caused by coherent wave interference. Speckle reduces contrast, obscures boundaries, and can make lesion interpretation more difficult for both clinicians and automated analysis systems.

Traditional denoising filters such as Gaussian smoothing, median filtering, and wavelet thresholding can reduce noise, but they often suppress fine anatomical detail at the same time. This trade-off is a major limitation in medical imaging, where preserving edges and lesion morphology is critical. Deep learning methods provide a stronger alternative because they can learn nonlinear mappings from noisy to clean images while retaining structural information.

Among deep learning architectures, U-Net is especially suitable for biomedical image restoration due to its encoder-decoder design and skip connections. The encoder captures contextual information, while the decoder reconstructs spatial detail. Skip connections help preserve low-level features that are important for lesion boundaries and texture structure.

Although denoising quality is important, reliability is also essential in medical applications. A model that produces a visually plausible output may still be uncertain in some regions, especially in the presence of ambiguous texture or heavy noise. For this reason, the proposed system incorporates Monte Carlo dropout to estimate predictive uncertainty. By performing multiple stochastic forward passes, the model can produce both a denoised image and a variance-based uncertainty map, which helps identify regions where the output should be interpreted with caution.

## II. Literature Survey
### A. Overview
Classical denoising techniques include Gaussian smoothing, Wiener filtering, median filtering, and wavelet shrinkage. These methods are simple and computationally efficient, but they are usually hand-designed and may over-smooth medically important structures. More advanced classical approaches such as BM3D improve performance by exploiting patch similarity, but they still rely on fixed priors and do not adapt well to all ultrasound conditions.

Deep learning methods, especially convolutional neural networks and U-Net variants, have achieved superior denoising performance by learning data-driven restoration mappings. These models can better preserve edges and anatomical patterns than classical filters. However, many existing denoising networks output only a single deterministic prediction and do not indicate how confident the model is in different regions of the image.

### B. Existing Models
The comparison in Table 1 summarizes representative methods discussed in the project context. Classical methods such as Gaussian filtering and BM3D provide baseline denoising capability, while CNN and U-Net models improve fidelity by learning from examples. The main limitation of most existing approaches is the lack of uncertainty estimation, which is especially relevant in medical imaging workflows.

#### Table 1. Model Comparison

| Method | Strengths | Limitations |
| --- | --- | --- |
| Gaussian Filter | Simple, fast, easy to implement | Blurs edges and removes detail |
| Wavelet Denoising | Better detail preservation than basic smoothing | Requires parameter tuning, limited adaptability |
| BM3D | Strong classical denoising performance | Computationally heavier, fixed priors |
| CNN | Learns nonlinear denoising mappings | Often deterministic, limited interpretability |
| U-Net | Preserves spatial detail through skip connections | Standard form does not quantify uncertainty |

## III. Methodology
### A. Preprocessing
Input ultrasound images are first converted to grayscale when necessary, resized to a uniform resolution, and normalized to the range used by the model. This standardization ensures consistent input dimensions and stabilizes training and inference.

### B. Noise Simulation
To emulate ultrasound degradation, speckle noise is simulated as multiplicative noise. This is more realistic than purely additive noise because speckle scales with local image intensity. The synthetic noise model allows controlled evaluation across different noise levels.

### C. U-Net Model
The denoising backbone is a U-Net encoder-decoder architecture. The encoder progressively extracts abstract features through convolution and downsampling, while the decoder restores the image through upsampling and feature fusion. Skip connections transfer high-resolution features from encoder stages to corresponding decoder stages, helping reconstruct sharp anatomical boundaries.

### D. Uncertainty Estimation
Monte Carlo dropout is used to estimate uncertainty at inference time. Dropout layers remain active during evaluation, and the model performs multiple forward passes for the same input. The mean of these predictions is used as the final denoised output, while the per-pixel variance across passes forms an uncertainty map. This map highlights areas where the model output is less stable, providing an additional reliability signal for clinical review.

## IV. System Architecture
The system is implemented as a full-stack application with three main components. The frontend is built with Streamlit and provides an interactive interface for image upload, visualization, and result inspection. The backend is implemented using Flask and exposes inference endpoints for deterministic denoising and uncertainty-aware prediction. The model itself is implemented in PyTorch.

The processing flow is:
Input image -> preprocessing -> U-Net inference -> optional Monte Carlo dropout sampling -> denoised output + uncertainty map.

#### Fig. 1. System Flow Diagram

```text
User Upload
   |
   v
Preprocessing
   |
   v
PyTorch U-Net Backend
   |
   +--> Deterministic Output
   |
   +--> Monte Carlo Dropout Sampling
            |
            +--> Mean Denoised Image
            +--> Uncertainty Map
   |
   v
Streamlit Visualization
```

#### Fig. 2. U-Net Structure

```text
Encoder -> Bottleneck -> Decoder
   |           |           |
   +----------- Skip Connections -----------+
```

## V. Experimental Results
### A. Dataset
The project is designed around the BUSI dataset for breast ultrasound enhancement experiments. The current repository includes image-based evaluation artifacts and robustness tables derived from BUSI-style ultrasound images. The workflow supports train/test splits and repeated evaluation under different noise settings.

### B. Metrics
Performance is measured using PSNR, SSIM, and MAE. PSNR captures reconstruction fidelity in decibels, SSIM measures structural similarity, and MAE reports the average absolute difference between the restored and reference images. Together, these metrics provide complementary evidence of denoising quality.

### C. Results
The recorded comparison table shows that the enhanced U-Net configuration outperforms the baseline MSE-trained U-Net. Specifically, PSNR improves from 20.04 dB to 27.02 dB and SSIM improves from 0.488 to 0.742 on the project comparison set. The robustness study further shows that the model maintains stronger performance under matched and moderate noise conditions, with the best recorded result at train noise 0.2 and test noise 0.2, where PSNR reaches 29.54 dB and SSIM reaches 0.851.

The uncertainty module produces pixel-level uncertainty maps that highlight ambiguous regions in noisy images. These maps are especially useful in areas with weak texture cues or stronger speckle corruption. In the current system, uncertainty is exposed through the backend inference endpoint and visualized in the Streamlit frontend alongside the restored image.

#### Table 2. Summary of Recorded Results

| Method | PSNR (dB) | SSIM | MAE |
| --- | --- | --- | --- |
| U-Net (MSE) | 20.04 | 0.488 | Not exported in current tables |
| U-Net + Hybrid + Noise | 27.02 | 0.742 | Not exported in current tables |

#### Table 3. Robustness Snapshot

| Train Noise | Test Noise | PSNR (dB) | SSIM |
| --- | --- | --- | --- |
| 0.2 | 0.2 | 29.54 | 0.851 |
| 0.2 | 0.4 | 25.41 | 0.673 |
| 0.2 | 0.6 | 21.08 | 0.495 |
| 0.4 | 0.4 | 27.02 | 0.742 |
| 0.6 | 0.6 | 25.07 | 0.671 |

## VI. Conclusion
This work demonstrates that U-Net-based denoising is effective for ultrasound image enhancement under speckle noise. The model improves image quality while preserving structure better than classical filtering methods. Monte Carlo dropout adds a second output in the form of uncertainty estimation, improving the reliability of the system by indicating where predictions are less certain. The combination of denoising and uncertainty visualization makes the proposed pipeline suitable for practical deployment in an interactive medical imaging workflow.

## VII. Future Work
Future extensions can strengthen the method in several directions. First, evaluation on larger and more diverse ultrasound datasets would improve generalization. Second, transformer-based architectures or hybrid CNN-transformer models may capture broader contextual information. Third, real-time optimization, quantization, and deployment on lower-resource hardware could make the system more suitable for clinical workflows.

## References
[1] U-Net for biomedical image segmentation and restoration literature.

[2] Monte Carlo dropout for approximate Bayesian uncertainty estimation.

[3] Ultrasound denoising and speckle reduction studies.

[4] BUSI dataset description and benchmark papers.

## Notes for Finalization
- Replace this draft reference list with the uploaded reference paper and any additional citations you want to match.
- If you want this converted into IEEE or Elsevier style, the section headings and references should be reformatted accordingly.
- If you want the paper tailored to a specific target conference or journal, the length, figure formatting, and citation style should be adjusted.