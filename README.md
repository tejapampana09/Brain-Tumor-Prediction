# 🧠 Dual AI Brain Tumor Classifier & Comparative Inference

An end-to-end Deep Learning system and interactive web application for detecting and classifying Brain Tumors from MRI scans into four distinct categories: **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor (Healthy)**.

This project features a **Side-by-Side Comparative AI Engine** hosting two distinct architectures running in parallel:
1. **Model A: MobileNetV2 (Transfer Learning)** trained using **TensorFlow / Keras 3**.
2. **Model B: Custom 6-Block ConvNet** trained from scratch using **PyTorch**.

---

## 🌟 Key Features

- **Side-by-Side Dual Inference**: Upload an MRI scan once and observe real-time predictions from both Keras and PyTorch models simultaneously.
- **Consensus & Divergence Detection**: Automatically flags whether both neural networks agree on the diagnosis or diverge.
- **Full Class Probability Distributions**: Visualized with animated, color-coded confidence meters.
- **Low Latency**: Lightweight MobileNetV2 and streamlined Custom CNN provide fast responses (~230ms - 280ms per scan).
- **Interactive Web UI**: Modern dark-mode interface supporting drag & drop uploads, live MRI previews, and 1-click test benchmarks.
- **CLI Utilities Included**: Ready-to-use standalone Python scripts for batch testing and single-image prediction.

---

## 📊 Model Architecture Comparison

| Feature | Model A: MobileNetV2 | Model B: Custom CNN |
| :--- | :--- | :--- |
| **Framework** | TensorFlow / Keras 3 (Torch Backend) | PyTorch |
| **Model Weights** | `Brain_Tumor_MobileNetV2.keras` | `custom_cnn_final.pt` |
| **Parameters** | ~2.4 Million | ~12.5 Million |
| **Input Shape** | 224 × 224 × 3 (RGB) | 200 × 200 × 1 (Grayscale) |
| **Preprocessing** | Built-in Rescaling ([-1, 1]) | Gaussian Blur + Normalization ([0, 1]) |
| **Architecture Design** | Pretrained Inverted Residual Bottlenecks + GAP | 6 ConvBlocks (Conv2d, BatchNorm, ReLU, MaxPool) + AdaptiveAvgPool |
| **Regularization** | Double Dropout (0.3) + Data Augmentation | Dropout (0.4) + BatchNorm |

---

## 🔬 Dataset & Classification Classes

Trained and evaluated on the benchmark **Brain Tumor MRI Dataset** consisting of T1-weighted Contrast Enhanced MRI slices:

1. **Glioma Tumor**: Tumors arising from glial tissue in the brain and spinal cord.
2. **Meningioma Tumor**: Typically benign tumors arising from the protective membranes surrounding the brain.
3. **No Tumor**: Healthy brain MRI scans with no detectable lesions.
4. **Pituitary Tumor**: Abnormal growth in the pituitary gland situated at the skull base.

---

## 🧹 Data Cleaning & Preprocessing Pipeline

Data quality and rigorous preprocessing are central to this project's reliability and generalizability:

### 1. Data Cleaning & Integrity Audit (`data_preprocessing.py`)
- **Corrupt File Elimination**: Every image is verified using PIL header inspection (`im.verify()`) with `LOAD_TRUNCATED_IMAGES = False` to discard broken or incomplete downloads.
- **Data Leakage Prevention via SHA-256 Hashing**: Medical MRI datasets frequently contain identical or nearly identical duplicate slices. All files are hashed with SHA-256; exact duplicate groups are kept within the same data split to ensure zero data leakage between training and testing.
- **Label Canonicalization**: Inconsistent directory names (`glioma_tumor`, `no_tumor`, etc.) are mapped into standardized canonical classes (`glioma`, `meningioma`, `notumor`, `pituitary`).

### 2. Dual Model Preprocessing Pipelines
Because each neural architecture was optimized for different visual representations, two distinct preprocessing pipelines run in parallel:

| Step | MobileNetV2 Pipeline | Custom CNN Pipeline |
| :--- | :--- | :--- |
| **Color Channels** | 3-Channel RGB (`im.convert("RGB")`) | 1-Channel Grayscale (`im.convert("L")`) |
| **Resolution** | 224 × 224 (Bilinear Interpolation) | 200 × 200 (Bilinear Interpolation) |
| **Noise Reduction** | Feature-level batch normalization | Gaussian Smoothing Blur (`kernel=3, sigma=0.5`) |
| **Normalization** | Rescaling to `[-1.0, 1.0]` | Min-Max Tensor Scaling to `[0.0, 1.0]` |

Run the dataset audit tool anytime via:
```bash
python data_preprocessing.py unseen_test_images
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/tejapampana09/Brain-Tumor-Prediction.git
cd Brain-Tumor-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
python -m uvicorn app:app --port 8000
```
Open your browser and navigate to:  
👉 **`http://127.0.0.1:8000`**

---

## 💻 CLI Usage

### Single Image Prediction
To predict a single MRI image from the command line:
```bash
python predict.py "path/to/mri_scan.jpg"
```

### Run Benchmark Test on Unseen MRI Scans
To evaluate the model suite against sample benchmark scans:
```bash
python test_unseen.py
```

---

## 📂 Project Structure

```text
Brain-Tumor-Prediction/
├── app.py                         # Full-stack FastAPI server + Responsive Dual UI
├── data_preprocessing.py          # Standalone dataset audit, cleaning & preprocessing pipeline
├── predict.py                     # CLI prediction script
├── test_unseen.py                 # Automated test suite for unseen benchmark images
├── Brain_Tumor_Classification_Complete_Walkthrough.ipynb # Step-by-step end-to-end ML training notebook
├── Brain_Tumor_MobileNetV2.keras   # Trained Keras MobileNetV2 model
├── custom_cnn_final.pt            # Trained PyTorch Custom CNN model
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules
├── README.md                      # Project documentation
└── unseen_test_images/            # Benchmark test scans for quick validation
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## ⚠️ Disclaimer

*This project is built strictly for educational, academic, and research purposes. It is not an FDA-approved clinical diagnostic instrument. Final medical diagnoses must always be determined by board-certified radiologists and medical professionals.*

---

## 👤 Author

Developed by **Teja Pampana**  
- GitHub: [@tejapampana09](https://github.com/tejapampana09)
