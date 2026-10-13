import os
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure backend
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from PIL import Image

MODEL_PATH = "Brain_Tumor_MobileNetV2.keras"
CLASSES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at '{image_path}'")
        return

    print(f"[*] Loading model from '{MODEL_PATH}'...")
    model = keras.models.load_model(MODEL_PATH)

    print(f"[*] Preprocessing image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Note: Model includes built-in rescaling_1 layer (scale: 1/127.5, offset: -1)
    # so raw pixel values [0, 255] are expected directly!

    print("[*] Running prediction...")
    preds = model.predict(img_batch, verbose=0)[0]

    print("\n" + "=" * 45)
    print(" PREDICTION RESULTS")
    print("=" * 45)
    for cls_name, prob in zip(CLASSES, preds):
        bar = "#" * int(prob * 25)
        print(f" {cls_name:<18}: {prob * 100:6.2f}% {bar}")
    
    top_idx = int(np.argmax(preds))
    print("=" * 45)
    print(f" [>>] Diagnosis: {CLASSES[top_idx]} ({preds[top_idx] * 100:.2f}% confidence)")
    print("=" * 45)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py test_brain_mri.jpg")
    else:
        predict_image(sys.argv[1])
