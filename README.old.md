<h1 class="code-line" data-line-start=0 data-line-end=1 ><a id="_UNetBased_Biomedical_Image_Enhancement_for_Ultrasound_Imaging_0"></a>🧠 U-Net–Based Biomedical Image Enhancement for Ultrasound Imaging</h1>
<hr>
<h2 class="code-line" data-line-start=4 data-line-end=5 ><a id="_Overview_4"></a>📌 Overview</h2>
<p class="has-line-data" data-line-start="6" data-line-end="7">This project presents a <strong>deep learning–based image processing system</strong> for enhancing <strong>ultrasound images</strong> by suppressing speckle noise and tissue clutter while preserving anatomical structures.</p>
<p class="has-line-data" data-line-start="8" data-line-end="9">A <strong>U-Net architecture</strong> is employed to learn an end-to-end enhancement mapping, and its performance is compared against a <strong>classical SVD-based filtering baseline</strong>. The system is deployed using a <strong>Flask backend</strong> and an interactive <strong>Streamlit frontend</strong> for real-time inference and visualization.</p>
<hr>
<h2 class="code-line" data-line-start=12 data-line-end=13 ><a id="_Objectives_12"></a>🎯 Objectives</h2>
<ul>
<li class="has-line-data" data-line-start="14" data-line-end="15">Enhance ultrasound image quality using deep learning</li>
<li class="has-line-data" data-line-start="15" data-line-end="16">Reduce speckle noise without distorting anatomical details</li>
<li class="has-line-data" data-line-start="16" data-line-end="17">Compare deep learning output with classical SVD filtering</li>
<li class="has-line-data" data-line-start="17" data-line-end="19">Provide a real-time, user-friendly interface for demonstration</li>
</ul>
<hr>
<h2 class="code-line" data-line-start=21 data-line-end=22 ><a id="_Domain_Details_21"></a>🧩 Domain Details</h2>
<ul>
<li class="has-line-data" data-line-start="23" data-line-end="24"><strong>Domain:</strong> Deep Learning</li>
<li class="has-line-data" data-line-start="24" data-line-end="25"><strong>Sub-Domain:</strong> Image Processing (Biomedical Imaging)</li>
<li class="has-line-data" data-line-start="25" data-line-end="26"><strong>Application Area:</strong> Ultrasound Image Enhancement</li>
<li class="has-line-data" data-line-start="26" data-line-end="28"><strong>Relevant SDG:</strong> SDG-3 – Good Health and Well-Being</li>
</ul>
<hr>
<h2 class="code-line" data-line-start=30 data-line-end=31 ><a id="_System_Architecture_30"></a>🏗️ System Architecture</h2>
<p class="has-line-data" data-line-start="32" data-line-end="43">Streamlit Frontend<br>
↓<br>
Flask REST API<br>
↓<br>
Preprocessing<br>
↓<br>
Trained U-Net Model<br>
↓<br>
Postprocessing + Metrics<br>
↓<br>
Enhanced Output</p>
<hr>
<h2 class="code-line" data-line-start=48 data-line-end=49 ><a id="_Model_Details_48"></a>🧠 Model Details</h2>
<ul>
<li class="has-line-data" data-line-start="50" data-line-end="51"><strong>Architecture:</strong> U-Net (Encoder–Decoder with Skip Connections)</li>
<li class="has-line-data" data-line-start="51" data-line-end="52"><strong>Input:</strong> Grayscale ultrasound image</li>
<li class="has-line-data" data-line-start="52" data-line-end="53"><strong>Output:</strong> Enhanced ultrasound image</li>
<li class="has-line-data" data-line-start="53" data-line-end="54"><strong>Loss Function:</strong> Mean Squared Error (MSE)</li>
<li class="has-line-data" data-line-start="54" data-line-end="55"><strong>Optimizer:</strong> Adam</li>
<li class="has-line-data" data-line-start="55" data-line-end="56"><strong>Training Mode:</strong> Autoencoder-style enhancement</li>
<li class="has-line-data" data-line-start="56" data-line-end="58"><strong>Acceleration:</strong> CUDA (if available)</li>
</ul>
<p class="has-line-data" data-line-start="58" data-line-end="59">A <strong>pre-trained model (<code>unet_model.pth</code>) is included</strong> to allow immediate inference without retraining.</p>
<hr>
<h2 class="code-line" data-line-start=62 data-line-end=63 ><a id="_Evaluation_Metrics_62"></a>📊 Evaluation Metrics</h2>
<p class="has-line-data" data-line-start="64" data-line-end="65">The system evaluates enhancement quality using:</p>
<ul>
<li class="has-line-data" data-line-start="66" data-line-end="67"><strong>PSNR</strong> (Peak Signal-to-Noise Ratio)</li>
<li class="has-line-data" data-line-start="67" data-line-end="68"><strong>SSIM</strong> (Structural Similarity Index)</li>
<li class="has-line-data" data-line-start="68" data-line-end="70"><strong>Runtime</strong> (Inference Time)</li>
</ul>
<h3 class="code-line" data-line-start=70 data-line-end=71 ><a id="Typical_Results_70"></a>Typical Results</h3>
<ul>
<li class="has-line-data" data-line-start="72" data-line-end="73"><strong>PSNR:</strong> ~43 dB</li>
<li class="has-line-data" data-line-start="73" data-line-end="74"><strong>SSIM:</strong> ~0.99</li>
<li class="has-line-data" data-line-start="74" data-line-end="76"><strong>Inference Time:</strong> ~0.3 seconds per image</li>
</ul>
<p class="has-line-data" data-line-start="76" data-line-end="77">These results indicate strong noise suppression while preserving anatomical structure.</p>
<hr>
<h2 class="code-line" data-line-start=79 data-line-end=80 ><a id="_Visual_Results_79"></a>🖼️ Visual Results</h2>
<h3 class="code-line" data-line-start=81 data-line-end=82 ><a id="Input_Ultrasound_Image_81"></a>Input Ultrasound Image</h3>
<p class="has-line-data" data-line-start="82" data-line-end="83"><img src="assets/screenshots/input.jpg" alt="Input Image"></p>
<h3 class="code-line" data-line-start=84 data-line-end=85 ><a id="UNet_Enhanced_Output_84"></a>U-Net Enhanced Output</h3>
<p class="has-line-data" data-line-start="85" data-line-end="86"><img src="assets/screenshots/unet_output.jpg" alt="U-Net Output"></p>
<h3 class="code-line" data-line-start=87 data-line-end=88 ><a id="SVDBased_Baseline_Output_87"></a>SVD-Based Baseline Output</h3>
<p class="has-line-data" data-line-start="88" data-line-end="89"><img src="assets/screenshots/svd_output.jpg" alt="SVD Output"></p>
<h3 class="code-line" data-line-start=90 data-line-end=91 ><a id="Quantitative_Evaluation_Metrics_90"></a>Quantitative Evaluation Metrics</h3>
<p class="has-line-data" data-line-start="91" data-line-end="92"><img src="assets/screenshots/metrics.png" alt="Metrics"></p>
<h2 class="code-line" data-line-start=93 data-line-end=94 ><a id="_Dataset_93"></a>📂 Dataset</h2>
<ul>
<li class="has-line-data" data-line-start="95" data-line-end="96"><strong>Dataset Used:</strong> BUSI – Breast Ultrasound Images Dataset</li>
<li class="has-line-data" data-line-start="96" data-line-end="98"><strong>Source:</strong> Publicly available research dataset</li>
</ul>
<p class="has-line-data" data-line-start="98" data-line-end="100"><strong>Note:</strong><br>
Since the task focuses on <strong>image enhancement rather than disease classification</strong>, the dataset is suitable irrespective of pathology type.</p>
<p class="has-line-data" data-line-start="101" data-line-end="102">Future work may evaluate generalization across other ultrasound modalities.</p>
<hr>
<h2 class="code-line" data-line-start=105 data-line-end=106 ><a id="_Project_Structure_105"></a>💻 Project Structure</h2>
<p class="has-line-data" data-line-start="107" data-line-end="134">u-net-biomedical-flow-project/<br>
├── backend/<br>
│ ├── <a href="http://app.py">app.py</a><br>
│ ├── <a href="http://inference.py">inference.py</a><br>
│ ├── model_loader.py<br>
│ └── models/<br>
│ └── unet_model.pth<br>
│<br>
├── frontend/<br>
│ └── streamlit_app.py<br>
│<br>
├── notebooks/<br>
│ ├── unet_training.ipynb<br>
│ ├── evaluation.ipynb<br>
│ └── data_generation.ipynb<br>
│<br>
├── utils/<br>
│ ├── <a href="http://preprocessing.py">preprocessing.py</a><br>
│ ├── <a href="http://postprocessing.py">postprocessing.py</a><br>
│ ├── <a href="http://metrics.py">metrics.py</a><br>
│ └── svd_baseline.py<br>
│<br>
├── data/<br>
│ └── images/<br>
│<br>
├── <a href="http://README.md">README.md</a><br>
└── .gitignore</p>
<hr>
<h2 class="code-line" data-line-start=139 data-line-end=140 ><a id="_How_to_Run_the_Project_139"></a>🚀 How to Run the Project</h2>
<h3 class="code-line" data-line-start=141 data-line-end=142 ><a id="1_Clone_the_Repository_141"></a>1️⃣ Clone the Repository</h3>
<pre><code class="has-line-data" data-line-start="143" data-line-end="164" class="language-bash">git <span class="hljs-built_in">clone</span> https://github.com/&lt;your-username&gt;/u-net-biomedical-flow-project.git
<span class="hljs-built_in">cd</span> u-net-biomedical-flow-project

<span class="hljs-number">2</span>️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate   <span class="hljs-comment"># Windows</span>

<span class="hljs-number">3</span>️⃣ Install Dependencies
pip install -r requirements.txt

<span class="hljs-number">4</span>️⃣ Run Backend (Flask API)
python -m backend.app

Expected output:
✅ Trained model loaded
Running on http://<span class="hljs-number">127.0</span>.<span class="hljs-number">0.1</span>:<span class="hljs-number">5000</span>

<span class="hljs-number">5</span>️⃣ Run Frontend (Streamlit)
streamlit run frontend/streamlit_app.py
Upload an ultrasound image to view enhancement results.
</code></pre>
<h1 class="code-line" data-line-start=166 data-line-end=167 ><a id="_Notebooks_166"></a>🧪 Notebooks</h1>
<p class="has-line-data" data-line-start="167" data-line-end="168"><strong>unet_training.ipynb</strong> – Model training and convergence</p>
<p class="has-line-data" data-line-start="169" data-line-end="170"><strong>evaluation.ipynb</strong> – Reserved for offline evaluation</p>
<p class="has-line-data" data-line-start="171" data-line-end="172"><strong>data_generation.ipynb</strong> – Reserved for data augmentation experiments</p>
<p class="has-line-data" data-line-start="173" data-line-end="174">The final evaluation is integrated directly into the deployed system.</p>
<h1 class="code-line" data-line-start=175 data-line-end=176 ><a id="_Key_Observations_175"></a>🧾 Key Observations</h1>
<p class="has-line-data" data-line-start="176" data-line-end="177">U-Net outperforms classical SVD filtering in visual quality</p>
<p class="has-line-data" data-line-start="178" data-line-end="179">Slight smoothing is expected and desirable in ultrasound enhancement</p>
<p class="has-line-data" data-line-start="180" data-line-end="181">High SSIM indicates successful structural preservation</p>
<h1 class="code-line" data-line-start=182 data-line-end=183 ><a id="_Future_Scope_182"></a>🔮 Future Scope</h1>
<p class="has-line-data" data-line-start="183" data-line-end="184">Evaluation on ultrasound images from different organs</p>
<p class="has-line-data" data-line-start="185" data-line-end="186">Edge-aware or perceptual loss for sharper outputs</p>
<p class="has-line-data" data-line-start="187" data-line-end="188">Real-time clinical integration</p>
<p class="has-line-data" data-line-start="189" data-line-end="190">Model compression for edge deployment</p>
<h1 class="code-line" data-line-start=191 data-line-end=192 ><a id="_Author_191"></a>👨‍💻 Author</h1>
<p class="has-line-data" data-line-start="192" data-line-end="195">Swayam Vijay Mehra<br>
Department of Conputational Intelligence (AIML)<br>
SRM Institute of Science and Technology</p>
<h1 class="code-line" data-line-start=196 data-line-end=197 ><a id="_License_196"></a>📜 License</h1>
<p class="has-line-data" data-line-start="197" data-line-end="198">This project is intended for academic and educational purposes only.</p>
<h1 class="code-line" data-line-start=199 data-line-end=200 ><a id="_Final_Note_199"></a>✅ Final Note</h1>
<p class="has-line-data" data-line-start="200" data-line-end="201">This repository contains a fully working, end-to-end deep learning system demonstrating the practical application of U-Net for biomedical image processing.</p>
