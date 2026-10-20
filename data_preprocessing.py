import os
import sys
import hashlib
from pathlib import Path
from PIL import Image, ImageFile
import numpy as np

# Prevent loading incomplete / truncated corrupt files
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==========================================
# 1. Class Aliases & Label Standardization
# ==========================================
CLASS_ALIASES = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "glioma tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "meningioma tumor": "meningioma",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "no tumor": "notumor",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "pituitary tumor": "pituitary",
}

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def canonical_class(folder_name):
    """Normalizes variations of folder names into standard canonical class names."""
    clean_name = folder_name.strip().lower().replace("-", "_")
    return CLASS_ALIASES.get(clean_name)

# ==========================================
# 2. Data Cleaning: Integrity & Deduplication
# ==========================================
def calculate_sha256(file_path):
    """Computes SHA-256 hash of an image file to detect exact duplicate scans."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_image_integrity(file_path):
    """
    Validates that an image file is not corrupt, unreadable, or truncated.
    Returns: (is_valid: bool, width, height, mode, error_message)
    """
    try:
        with Image.open(file_path) as im:
            im.verify()  # Verifies image header and integrity without full decoding
        with Image.open(file_path) as im:
            im.load()    # Ensures image blocks decode cleanly
            return True, im.width, im.height, im.mode, None
    except Exception as e:
        return False, None, None, None, str(e)

def audit_dataset_directory(directory_path):
    """
    Scans, audits, cleans, and deduplicates an MRI image dataset directory.
    - Filters out corrupt or unreadable files.
    - Flags exact duplicates using SHA-256 hashes to prevent data leakage.
    - Summarizes class distribution and image resolutions.
    """
    root = Path(directory_path)
    if not root.exists():
        print(f"[-] Directory not found: {directory_path}")
        return None

    print("=" * 70)
    print(f"🔍 AUDITING & CLEANING DATASET: {root.resolve()}")
    print("=" * 70)

    valid_records = []
    corrupt_files = []
    hashes = {}
    duplicates = []

    all_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    print(f"[*] Found {len(all_files)} total image candidate files.\n")

    for p in all_files:
        is_valid, w, h, mode, err = verify_image_integrity(p)
        if not is_valid:
            corrupt_files.append({"path": str(p), "error": err})
            continue

        file_hash = calculate_sha256(p)
        if file_hash in hashes:
            duplicates.append({"original": hashes[file_hash], "duplicate": str(p), "hash": file_hash})
        else:
            hashes[file_hash] = str(p)

        parent_class = canonical_class(p.parent.name) or canonical_class(p.parent.parent.name) or "unlabeled"
        valid_records.append({
            "path": str(p),
            "class": parent_class,
            "width": w,
            "height": h,
            "mode": mode,
            "hash": file_hash
        })

    # Summary Report
    print(f"[+] Clean, Valid Images : {len(valid_records)}")
    print(f"[-] Corrupt/Damaged Files : {len(corrupt_files)}")
    print(f"[!] Exact Duplicate Scans: {len(duplicates)}")

    if corrupt_files:
        print("\n--- Corrupt Files Detected ---")
        for c in corrupt_files[:5]:
            print(f"  ❌ {c['path']} -> {c['error']}")

    if duplicates:
        print("\n--- Duplicate Scans (Data Leakage Prevention) ---")
        for d in duplicates[:5]:
            print(f"  ⚠️ Duplicate: {os.path.basename(d['duplicate'])} == {os.path.basename(d['original'])}")

    class_counts = {}
    for r in valid_records:
        cls = r["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print("\n--- Cleaned Class Distribution ---")
    for cls, count in class_counts.items():
        print(f"  - {cls:<16}: {count} scans")

    print("=" * 70)
    return valid_records

# ==========================================
# 3. Model Preprocessing Pipelines
# ==========================================
def preprocess_for_mobilenet(image_path_or_pil):
    """
    Preprocessing pipeline for MobileNetV2 (Keras):
    1. Read and decode image into RGB color space (3 channels).
    2. Bilinear resize to 224x224 pixels.
    3. Convert to float32 NumPy array with shape (1, 224, 224, 3).
    4. Note: Rescaling to [-1, 1] is performed via the model's built-in Rescaling layer.
    """
    if isinstance(image_path_or_pil, (str, Path)):
        img = Image.open(image_path_or_pil)
    else:
        img = image_path_or_pil

    img_rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img_rgb, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0) # Shape: (1, 224, 224, 3)
    return batch

def preprocess_for_custom_cnn(image_path_or_pil):
    """
    Preprocessing pipeline for Custom CNN (PyTorch):
    1. Convert image to single-channel Grayscale (mode 'L').
    2. Resize to 200x200 pixels using Bilinear interpolation.
    3. Apply Gaussian Smoothing (kernel=3, sigma=0.5) to remove high-frequency scanner noise.
    4. Normalize pixel values from [0, 255] to [0.0, 1.0] as a PyTorch Tensor of shape (1, 1, 200, 200).
    """
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if isinstance(image_path_or_pil, (str, Path)):
        img = Image.open(image_path_or_pil)
    else:
        img = image_path_or_pil

    img_gray = img.convert("L")
    tf = transforms.Compose([
        transforms.Resize((200, 200), interpolation=InterpolationMode.BILINEAR),
        transforms.GaussianBlur(kernel_size=3, sigma=0.5),
        transforms.ToTensor(), # Scales [0, 255] -> [0.0, 1.0]
    ])
    tensor_batch = tf(img_gray).unsqueeze(0) # Shape: (1, 1, 200, 200)
    return tensor_batch

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "unseen_test_images"
    print(f"Running data cleaning & audit on '{target}'...")
    audit_dataset_directory(target)
