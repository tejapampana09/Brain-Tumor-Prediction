import os
import sys

# Ensure UTF-8 output encoding on Windows console
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
from PIL import Image
import numpy as np

MODEL_KERAS_PATH = "Brain_Tumor_MobileNetV2.keras"
MODEL_PYTORCH_PATH = "custom_cnn_final.pt"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor (Healthy)",
    "pituitary": "Pituitary Tumor"
}

# ==========================================
# PyTorch CustomCNN Definition & Transforms
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
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
    transforms.GaussianBlur(kernel_size=3, sigma=0.5),
    transforms.ToTensor(),
])

def predict_dual(image_path):
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at '{image_path}'")
        return

    print("=" * 65)
    print(f"🧠 DUAL AI BRAIN TUMOR PREDICTION: {os.path.basename(image_path)}")
    print("=" * 65)

    pil_raw = Image.open(image_path)

    # ---------------------------------------------
    # 1. Model A: MobileNetV2 (Keras)
    # ---------------------------------------------
    print("\n[*] [Model 1: MobileNetV2 (Keras Transfer Learning)]")
    if os.path.exists(MODEL_KERAS_PATH):
        mobilenet = keras.models.load_model(MODEL_KERAS_PATH)
        img_rgb = pil_raw.convert("RGB").resize((224, 224))
        arr_rgb = np.expand_dims(np.array(img_rgb, dtype=np.float32), axis=0)
        m1_preds = mobilenet.predict(arr_rgb, verbose=0)[0]
        
        m1_top = int(np.argmax(m1_preds))
        m1_class = CLASS_NAMES[m1_top]
        m1_conf = float(m1_preds[m1_top]) * 100

        print(f"    Diagnosis  : {DISPLAY_NAMES[m1_class]} ({m1_conf:.2f}%)")
        print("    Confidence Breakdown:")
        for idx, cls in enumerate(CLASS_NAMES):
            bar = "#" * int(m1_preds[idx] * 20)
            print(f"      - {DISPLAY_NAMES[cls]:<20}: {m1_preds[idx] * 100:6.2f}% {bar}")
    else:
        print(f"[-] MobileNetV2 weights not found at '{MODEL_KERAS_PATH}'")
        m1_class, m1_conf = None, 0.0

    # ---------------------------------------------
    # 2. Model B: Custom CNN (PyTorch)
    # ---------------------------------------------
    print("\n[*] [Model 2: Custom CNN (PyTorch 6-Block ConvNet)]")
    if os.path.exists(MODEL_PYTORCH_PATH):
        cnn_model = CustomCNN()
        ckpt = torch.load(MODEL_PYTORCH_PATH, map_location='cpu')
        cnn_model.load_state_dict(ckpt['model_state_dict'])
        cnn_model.eval()

        img_gray = pil_raw.convert("L")
        tensor_in = pytorch_eval_tf(img_gray).unsqueeze(0)
        with torch.no_grad():
            logits = cnn_model(tensor_in)
            m2_preds = torch.softmax(logits, dim=1)[0].numpy()

        m2_top = int(np.argmax(m2_preds))
        m2_class = CLASS_NAMES[m2_top]
        m2_conf = float(m2_preds[m2_top]) * 100

        print(f"    Diagnosis  : {DISPLAY_NAMES[m2_class]} ({m2_conf:.2f}%)")
        print("    Confidence Breakdown:")
        for idx, cls in enumerate(CLASS_NAMES):
            bar = "#" * int(m2_preds[idx] * 20)
            print(f"      - {DISPLAY_NAMES[cls]:<20}: {m2_preds[idx] * 100:6.2f}% {bar}")
    else:
        print(f"[-] Custom CNN checkpoint not found at '{MODEL_PYTORCH_PATH}'")
        m2_class, m2_conf = None, 0.0

    # ---------------------------------------------
    # 3. Consensus Summary
    # ---------------------------------------------
    print("\n" + "=" * 65)
    if m1_class and m2_class:
        if m1_class == m2_class:
            print(f"[CONSENSUS: AGREE] Both models agree on: {DISPLAY_NAMES[m1_class].upper()}")
        else:
            print(f"[CONSENSUS: DIVERGE] MobileNetV2: {DISPLAY_NAMES[m1_class]} vs Custom CNN: {DISPLAY_NAMES[m2_class]}")
    print("=" * 65)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_mri_image>")
        print("Example: python predict.py unseen_test_images/glioma/glioma_1.jpg")
    else:
        predict_dual(sys.argv[1])
