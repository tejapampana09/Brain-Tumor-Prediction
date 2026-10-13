import os
import io
import sys
import numpy as np
from PIL import Image

# Ensure UTF-8 output encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure Keras backend
os.environ["KERAS_BACKEND"] = "torch"
import keras

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

MODEL_KERAS_PATH = "Brain_Tumor_MobileNetV2.keras"
MODEL_PYTORCH_PATH = "custom_cnn_final.pt"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor (Healthy)",
    "pituitary": "Pituitary Tumor"
}

CLASS_DESCRIPTIONS = {
    "glioma": "Arising from glial cells in the brain/spinal cord.",
    "meningioma": "Arising from the protective meninges membranes.",
    "notumor": "No tumor or abnormal lesion detected.",
    "pituitary": "Abnormal growth in the pituitary gland at the skull base."
}

# ==========================================
# 1. PyTorch CustomCNN Definition & Loader
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.block(x)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 512)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

pytorch_eval_tf = transforms.Compose([
    transforms.Resize((200, 200), interpolation=InterpolationMode.BILINEAR),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
])

# Load models
print("[*] Loading Model 1: MobileNetV2 (Keras)...")
try:
    mobilenet_model = keras.models.load_model(MODEL_KERAS_PATH)
    print("[+] MobileNetV2 loaded successfully!")
except Exception as e:
    print(f"[-] Failed to load MobileNetV2: {e}")
    mobilenet_model = None

print("[*] Loading Model 2: CustomCNN (PyTorch)...")
try:
    custom_cnn_model = CustomCNN()
    if os.path.exists(MODEL_PYTORCH_PATH):
        ckpt = torch.load(MODEL_PYTORCH_PATH, map_location='cpu')
        custom_cnn_model.load_state_dict(ckpt['model_state_dict'])
        custom_cnn_model.eval()
        print("[+] CustomCNN loaded successfully!")
    else:
        print(f"[-] PyTorch checkpoint not found at {MODEL_PYTORCH_PATH}")
        custom_cnn_model = None
except Exception as e:
    print(f"[-] Failed to load CustomCNN: {e}")
    custom_cnn_model = None

app = FastAPI(title="Dual Brain Tumor AI Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dual AI Brain Tumor Comparison</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: #1f2937;
            --accent-blue: #3b82f6;
            --accent-indigo: #6366f1;
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --border-color: #374151;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 1100px;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .badges-row {
            display: flex;
            justify-content: center;
            gap: 0.75rem;
            margin-bottom: 0.8rem;
        }

        .badge {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 9999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .badge.pytorch {
            background: rgba(168, 85, 247, 0.15);
            color: #c084fc;
            border-color: rgba(168, 85, 247, 0.3);
        }

        header h1 {
            font-size: 2.3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #38bdf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.05rem;
        }

        .main-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 1.25rem;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
        }

        /* Samples Bar */
        .samples-bar {
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.02);
            border: 1px dashed var(--border-color);
            border-radius: 0.75rem;
        }

        .samples-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }

        .sample-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }

        .sample-btn {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.5rem 0.9rem;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .sample-btn:hover {
            border-color: var(--accent-blue);
            background: rgba(59, 130, 246, 0.15);
            transform: translateY(-1px);
        }

        /* Upload Area */
        .upload-area {
            border: 2px dashed var(--border-color);
            border-radius: 1rem;
            padding: 2rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: rgba(31, 41, 55, 0.4);
        }

        .upload-area:hover, .upload-area.dragover {
            border-color: var(--accent-blue);
            background: rgba(59, 130, 246, 0.08);
        }

        .upload-icon {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            display: block;
        }

        .upload-text {
            font-size: 1.05rem;
            font-weight: 500;
            margin-bottom: 0.25rem;
        }

        .upload-sub {
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        #file-input {
            display: none;
        }

        /* Preview Image */
        .preview-wrapper {
            display: none;
            text-align: center;
            margin-top: 1.5rem;
        }

        .preview-img {
            max-width: 220px;
            max-height: 220px;
            border-radius: 0.75rem;
            border: 2px solid var(--border-color);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        }

        /* Consensus Banner */
        .consensus-banner {
            display: none;
            margin-top: 1.5rem;
            padding: 1rem 1.25rem;
            border-radius: 0.75rem;
            font-weight: 600;
            text-align: center;
            font-size: 1.05rem;
        }

        .consensus-banner.agree {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399;
        }

        .consensus-banner.diverge {
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.4);
            color: #fbbf24;
        }

        /* Dual Results Grid */
        .dual-results {
            display: none;
            margin-top: 1.5rem;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 820px) {
            .dual-results {
                grid-template-columns: 1fr;
            }
        }

        .model-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }

        .model-tag {
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 0.375rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .tag-keras {
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .tag-pytorch {
            background: rgba(168, 85, 247, 0.2);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.3);
        }

        .prediction-hero {
            padding: 1rem;
            border-radius: 0.75rem;
            margin-bottom: 1.25rem;
            text-align: center;
        }

        .prediction-hero.tumor {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .prediction-hero.healthy {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .hero-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }

        .hero-class {
            font-size: 1.4rem;
            font-weight: 700;
        }

        .hero-conf {
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }

        .prob-row {
            margin-bottom: 0.85rem;
        }

        .prob-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 0.3rem;
        }

        .progress-track {
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.5s ease;
        }

        .fill-mobilenet {
            background: linear-gradient(90deg, #3b82f6, #06b6d4);
        }

        .fill-customcnn {
            background: linear-gradient(90deg, #a855f7, #ec4899);
        }

        /* Tech Specs Table */
        .specs-table {
            width: 100%;
            margin-top: 1.5rem;
            border-collapse: collapse;
            font-size: 0.8rem;
            color: var(--text-muted);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            overflow: hidden;
        }

        .specs-table th, .specs-table td {
            padding: 0.6rem 0.8rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        .specs-table th {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-main);
        }

        /* Spinner */
        .spinner {
            display: none;
            width: 44px;
            height: 44px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top-color: var(--accent-blue);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 2rem auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            margin-top: 2rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badges-row">
                <span class="badge">Model A • MobileNetV2 (Keras)</span>
                <span class="badge pytorch">Model B • Custom CNN (PyTorch)</span>
            </div>
            <h1>Dual Brain Tumor AI Classifier</h1>
            <p>Upload a Brain MRI scan to see real-time inference from both Deep Learning models side-by-side</p>
        </header>

        <div class="main-card">
            <!-- Sample Test Scans -->
            <div class="samples-bar">
                <div class="samples-title">⚡ Quick Test With Benchmark MRI Scans:</div>
                <div class="sample-buttons">
                    <button class="sample-btn" onclick="testSample('glioma', 'glioma_1.jpg')">🧪 Glioma Sample</button>
                    <button class="sample-btn" onclick="testSample('meningioma', 'meningioma_1.jpg')">🧪 Meningioma Sample</button>
                    <button class="sample-btn" onclick="testSample('notumor', 'notumor_1.jpg')">🧪 Normal (No Tumor)</button>
                    <button class="sample-btn" onclick="testSample('pituitary', 'pituitary_1.jpg')">🧪 Pituitary Sample</button>
                </div>
            </div>

            <!-- Drag & Drop Upload -->
            <div class="upload-area" id="drop-area" onclick="document.getElementById('file-input').click()">
                <span class="upload-icon">🧠</span>
                <div class="upload-text">Click to browse or Drag & Drop an MRI scan</div>
                <div class="upload-sub">Supports JPG, PNG, JPEG formats</div>
                <input type="file" id="file-input" accept="image/*" onchange="handleFileSelect(event)">
            </div>

            <div class="preview-wrapper" id="preview-wrapper">
                <img id="preview-image" class="preview-img" alt="MRI Scan Preview">
                <div id="preview-filename" style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-muted);"></div>
            </div>

            <div class="spinner" id="spinner"></div>

            <!-- Consensus Banner -->
            <div class="consensus-banner" id="consensus-banner"></div>

            <!-- Dual Prediction Cards -->
            <div class="dual-results" id="dual-results">
                <!-- Model 1: MobileNetV2 -->
                <div class="model-panel">
                    <div class="panel-header">
                        <div>
                            <h3 style="font-size: 1.1rem; font-weight: 700;">MobileNetV2</h3>
                            <div style="font-size: 0.75rem; color: var(--text-muted);">Transfer Learning • 2.4M params</div>
                        </div>
                        <span class="model-tag tag-keras">Keras / TF</span>
                    </div>

                    <div class="prediction-hero" id="m1-hero">
                        <div class="hero-label">Diagnosis</div>
                        <div class="hero-class" id="m1-class">---</div>
                        <div class="hero-conf" id="m1-conf">--%</div>
                    </div>

                    <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 600; color: var(--text-muted); margin-bottom: 0.75rem;">
                        Probabilities:
                    </div>
                    <div id="m1-probs"></div>

                    <table class="specs-table">
                        <tr><td>Input Size</td><td>224 × 224 (RGB)</td></tr>
                        <tr><td>Normalization</td><td>Built-in Rescaling ([-1, 1])</td></tr>
                        <tr><td>Latency</td><td id="m1-time">-- ms</td></tr>
                    </table>
                </div>

                <!-- Model 2: Custom CNN -->
                <div class="model-panel">
                    <div class="panel-header">
                        <div>
                            <h3 style="font-size: 1.1rem; font-weight: 700;">Custom CNN</h3>
                            <div style="font-size: 0.75rem; color: var(--text-muted);">6-Block Deep ConvNet • 12.5M params</div>
                        </div>
                        <span class="model-tag tag-pytorch">PyTorch</span>
                    </div>

                    <div class="prediction-hero" id="m2-hero">
                        <div class="hero-label">Diagnosis</div>
                        <div class="hero-class" id="m2-class">---</div>
                        <div class="hero-conf" id="m2-conf">--%</div>
                    </div>

                    <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 600; color: var(--text-muted); margin-bottom: 0.75rem;">
                        Probabilities:
                    </div>
                    <div id="m2-probs"></div>

                    <table class="specs-table">
                        <tr><td>Input Size</td><td>200 × 200 (Grayscale)</td></tr>
                        <tr><td>Preprocessing</td><td>Gaussian Blur + [0, 1]</td></tr>
                        <tr><td>Latency</td><td id="m2-time">-- ms</td></tr>
                    </table>
                </div>
            </div>
        </div>

        <footer>
            Comparative Analysis: Keras MobileNetV2 vs. PyTorch CustomCNN • Research & Educational Purposes Only
        </footer>
    </div>

    <script>
        const dropArea = document.getElementById('drop-area');
        const spinner = document.getElementById('spinner');
        const previewWrapper = document.getElementById('preview-wrapper');
        const previewImage = document.getElementById('preview-image');
        const previewFilename = document.getElementById('preview-filename');
        const consensusBanner = document.getElementById('consensus-banner');
        const dualResults = document.getElementById('dual-results');

        ['dragenter', 'dragover'].forEach(name => {
            dropArea.addEventListener(name, (e) => {
                e.preventDefault();
                dropArea.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropArea.addEventListener(name, (e) => {
                e.preventDefault();
                dropArea.classList.remove('dragover');
            });
        });

        dropArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadAndPredict(files[0]);
            }
        });

        function handleFileSelect(e) {
            if (e.target.files.length > 0) {
                uploadAndPredict(e.target.files[0]);
            }
        }

        async function testSample(category, filename) {
            const sampleUrl = `/api/sample/${category}/${filename}`;
            spinner.style.display = 'block';
            consensusBanner.style.display = 'none';
            dualResults.style.display = 'none';

            try {
                const response = await fetch(sampleUrl);
                const blob = await response.blob();
                const file = new File([blob], filename, { type: 'image/jpeg' });
                await uploadAndPredict(file);
            } catch (err) {
                alert('Error loading sample: ' + err);
                spinner.style.display = 'none';
            }
        }

        async function uploadAndPredict(file) {
            previewImage.src = URL.createObjectURL(file);
            previewFilename.innerText = file.name;
            previewWrapper.style.display = 'block';

            spinner.style.display = 'block';
            consensusBanner.style.display = 'none';
            dualResults.style.display = 'none';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) {
                    throw new Error('Prediction failed with status: ' + res.status);
                }

                const data = await res.json();
                renderComparison(data);
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                spinner.style.display = 'none';
            }
        }

        function renderComparison(data) {
            const m1 = data.mobilenet;
            const m2 = data.custom_cnn;

            // Render Model 1 (MobileNetV2)
            document.getElementById('m1-class').innerText = m1.display_name;
            document.getElementById('m1-conf').innerText = m1.confidence.toFixed(2) + '%';
            document.getElementById('m1-time').innerText = m1.latency_ms + ' ms';
            document.getElementById('m1-hero').className = 'prediction-hero ' + (m1.prediction === 'notumor' ? 'healthy' : 'tumor');

            const m1Probs = document.getElementById('m1-probs');
            m1Probs.innerHTML = '';
            m1.probabilities.forEach(p => {
                m1Probs.innerHTML += `
                    <div class="prob-row">
                        <div class="prob-meta">
                            <span>${p.display_name}</span>
                            <span style="font-weight: 600; color: #60a5fa;">${p.percentage.toFixed(2)}%</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill fill-mobilenet" style="width: ${Math.max(p.percentage, 1)}%"></div>
                        </div>
                    </div>
                `;
            });

            // Render Model 2 (Custom CNN)
            document.getElementById('m2-class').innerText = m2.display_name;
            document.getElementById('m2-conf').innerText = m2.confidence.toFixed(2) + '%';
            document.getElementById('m2-time').innerText = m2.latency_ms + ' ms';
            document.getElementById('m2-hero').className = 'prediction-hero ' + (m2.prediction === 'notumor' ? 'healthy' : 'tumor');

            const m2Probs = document.getElementById('m2-probs');
            m2Probs.innerHTML = '';
            m2.probabilities.forEach(p => {
                m2Probs.innerHTML += `
                    <div class="prob-row">
                        <div class="prob-meta">
                            <span>${p.display_name}</span>
                            <span style="font-weight: 600; color: #c084fc;">${p.percentage.toFixed(2)}%</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill fill-customcnn" style="width: ${Math.max(p.percentage, 1)}%"></div>
                        </div>
                    </div>
                `;
            });

            // Consensus banner
            if (m1.prediction === m2.prediction) {
                consensusBanner.className = 'consensus-banner agree';
                consensusBanner.innerHTML = `✅ Consensus: Both Models Agree on <span style="text-decoration: underline;">${m1.display_name}</span>`;
            } else {
                consensusBanner.className = 'consensus-banner diverge';
                consensusBanner.innerHTML = `⚠️ Divergence: MobileNetV2 predicts <strong>${m1.display_name}</strong> while Custom CNN predicts <strong>${m2.display_name}</strong>`;
            }

            consensusBanner.style.display = 'block';
            dualResults.style.display = 'grid';
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_index():
    return HTML_PAGE

@app.get("/api/sample/{category}/{filename}")
def get_sample(category: str, filename: str):
    path = os.path.join("unseen_test_images", category, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(path)

@app.post("/api/predict")
async def predict_dual(file: UploadFile = File(...)):
    if mobilenet_model is None or custom_cnn_model is None:
        raise HTTPException(status_code=500, detail="One or both models failed to load.")

    try:
        import time
        content = await file.read()
        pil_raw = Image.open(io.BytesIO(content))

        # -----------------------------
        # 1. Inference: MobileNetV2 (RGB, 224x224)
        # -----------------------------
        t0 = time.time()
        img_rgb = pil_raw.convert("RGB").resize((224, 224))
        arr_rgb = np.expand_dims(np.array(img_rgb, dtype=np.float32), axis=0)
        m1_preds = mobilenet_model.predict(arr_rgb, verbose=0)[0]
        m1_time = round((time.time() - t0) * 1000, 1)

        m1_top_idx = int(np.argmax(m1_preds))
        m1_top_class = CLASS_NAMES[m1_top_idx]
        m1_conf = float(m1_preds[m1_top_idx]) * 100

        m1_probs = []
        for idx, cls in enumerate(CLASS_NAMES):
            m1_probs.append({
                "class_name": cls,
                "display_name": DISPLAY_NAMES[cls],
                "probability": float(m1_preds[idx]),
                "percentage": float(m1_preds[idx]) * 100
            })
        m1_probs.sort(key=lambda x: x["probability"], reverse=True)

        # -----------------------------
        # 2. Inference: Custom CNN (Grayscale, 200x200, PyTorch)
        # -----------------------------
        t0 = time.time()
        img_gray = pil_raw.convert("L")
        tensor_in = pytorch_eval_tf(img_gray).unsqueeze(0)
        with torch.no_grad():
            out_logits = custom_cnn_model(tensor_in)
            m2_preds = torch.softmax(out_logits, dim=1)[0].numpy()
        m2_time = round((time.time() - t0) * 1000, 1)

        m2_top_idx = int(np.argmax(m2_preds))
        m2_top_class = CLASS_NAMES[m2_top_idx]
        m2_conf = float(m2_preds[m2_top_idx]) * 100

        m2_probs = []
        for idx, cls in enumerate(CLASS_NAMES):
            m2_probs.append({
                "class_name": cls,
                "display_name": DISPLAY_NAMES[cls],
                "probability": float(m2_preds[idx]),
                "percentage": float(m2_preds[idx]) * 100
            })
        m2_probs.sort(key=lambda x: x["probability"], reverse=True)

        return JSONResponse({
            "mobilenet": {
                "prediction": m1_top_class,
                "display_name": DISPLAY_NAMES[m1_top_class],
                "confidence": m1_conf,
                "description": CLASS_DESCRIPTIONS[m1_top_class],
                "probabilities": m1_probs,
                "latency_ms": m1_time
            },
            "custom_cnn": {
                "prediction": m2_top_class,
                "display_name": DISPLAY_NAMES[m2_top_class],
                "confidence": m2_conf,
                "description": CLASS_DESCRIPTIONS[m2_top_class],
                "probabilities": m2_probs,
                "latency_ms": m2_time
            }
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
