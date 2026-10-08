import cv2
import os

from src.preprocessing import preprocess_image
from src.enhancement import (
    histogram_equalization,
    clahe_enhancement,
    hybrid_he_clahe,
    adaptive_hybrid_he_clahe
)

from src.metrics import calculate_metrics


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "dataset/xray/knee.png"

OUTPUT_DIR = "results/new_algorithm_test"


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    raise FileNotFoundError(
        f"Could not load: {IMAGE_PATH}"
    )


# ============================================================
# PREPROCESS
# ============================================================

original = preprocess_image(
    image
)

print(
    "Image size:",
    original.shape
)


# ============================================================
# RUN METHODS
# ============================================================

print()
print("Running HE...")

he = histogram_equalization(
    original
)


print("Running CLAHE...")

clahe = clahe_enhancement(
    original,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
)


print("Running baseline Hybrid...")

hybrid = hybrid_he_clahe(
    original,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
)


print(
    "Running paper-inspired Adaptive Hybrid..."
)

adaptive = adaptive_hybrid_he_clahe(
    original,
    min_clip=1.0,
    max_clip=2.0,
    grid_size=8
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE IMAGES
# ============================================================

cv2.imwrite(
    f"{OUTPUT_DIR}/original.png",
    original
)

cv2.imwrite(
    f"{OUTPUT_DIR}/he.png",
    he
)

cv2.imwrite(
    f"{OUTPUT_DIR}/clahe.png",
    clahe
)

cv2.imwrite(
    f"{OUTPUT_DIR}/hybrid.png",
    hybrid
)

cv2.imwrite(
    f"{OUTPUT_DIR}/adaptive_hybrid.png",
    adaptive
)


# ============================================================
# CALCULATE METRICS
# ============================================================

methods = {

    "HE":
        he,

    "CLAHE":
        clahe,

    "Hybrid HE-CLAHE":
        hybrid,

    "Adaptive Hybrid":
        adaptive
}


print()
print("=" * 70)
print("KNEE X-RAY RESULTS")
print("=" * 70)

print(
    f"{'Method':<25}"
    f"{'MSE':>12}"
    f"{'PSNR':>12}"
    f"{'SSIM':>12}"
)

print("-" * 70)


for name, result in methods.items():

    metrics = calculate_metrics(
        original,
        result
    )

    print(
        f"{name:<25}"
        f"{metrics['MSE']:>12.4f}"
        f"{metrics['PSNR']:>12.4f}"
        f"{metrics['SSIM']:>12.4f}"
    )


print("-" * 70)

print()
print(
    "Images saved to:"
)

print(
    OUTPUT_DIR
)