import os
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from PIL import Image

MODEL_PATH = "Brain_Tumor_MobileNetV2.keras"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor (Healthy)",
    "pituitary": "Pituitary Tumor"
}

def evaluate_unseen_suite():
    print("=" * 70)
    print("  EVALUATING MODEL ON 8 UNSEEN BENCHMARK TEST MRI IMAGES")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return

    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    
    test_suite = [
        ("glioma", "unseen_test_images/glioma/glioma_1.jpg"),
        ("glioma", "unseen_test_images/glioma/glioma_2.jpg"),
        ("meningioma", "unseen_test_images/meningioma/meningioma_1.jpg"),
        ("meningioma", "unseen_test_images/meningioma/meningioma_2.jpg"),
        ("notumor", "unseen_test_images/notumor/notumor_1.jpg"),
        ("notumor", "unseen_test_images/notumor/notumor_2.jpg"),
        ("pituitary", "unseen_test_images/pituitary/pituitary_1.jpg"),
        ("pituitary", "unseen_test_images/pituitary/pituitary_2.jpg"),
    ]

    correct_count = 0
    total_count = len(test_suite)

    print(f"\nEvaluating {total_count} unseen test samples...\n")
    print(f"{'#':<3} {'Image File':<32} {'True Label':<16} {'Predicted':<16} {'Confidence':<10} {'Result'}")
    print("-" * 85)

    results_detail = []

    for idx, (true_label, img_path) in enumerate(test_suite, 1):
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB").resize((224, 224))
        batch = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

        probs = model.predict(batch, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[top_idx]
        confidence = float(probs[top_idx]) * 100

        is_correct = (pred_label == true_label)
        if is_correct:
            correct_count += 1
            result_mark = "[CORRECT]"
        else:
            result_mark = "[WRONG]"

        filename_short = os.path.basename(img_path)
        print(f"{idx:<3} {img_path:<32} {true_label:<16} {pred_label:<16} {confidence:6.2f}%    {result_mark}")
        results_detail.append((img_path, true_label, pred_label, confidence, probs, is_correct))

    accuracy = (correct_count / total_count) * 100
    print("-" * 85)
    print(f"\nFinal Score: {correct_count}/{total_count} Correct ({accuracy:.1f}% Accuracy)")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_unseen_suite()
