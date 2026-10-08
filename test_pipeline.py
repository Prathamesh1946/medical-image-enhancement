import cv2

from src.preprocessing import preprocess_image
from src.enhancement import (
    histogram_equalization,
    clahe_enhancement,
    hybrid_he_clahe,
    adaptive_hybrid_he_clahe
)

from src.metrics import calculate_metrics


IMAGE_PATH = "dataset/xray/knee.png"


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# ------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------

gray = preprocess_image(
    image
)

print(
    "Preprocessed image:",
    gray.shape
)


# ------------------------------------------------------------
# Algorithms
# ------------------------------------------------------------

he = histogram_equalization(
    gray
)

clahe = clahe_enhancement(
    gray
)

baseline_hybrid = hybrid_he_clahe(
    gray
)

adaptive_hybrid = adaptive_hybrid_he_clahe(
    gray
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

results = {

    "HE":
        calculate_metrics(
            gray,
            he
        ),

    "CLAHE":
        calculate_metrics(
            gray,
            clahe
        ),

    "Baseline Hybrid":
        calculate_metrics(
            gray,
            baseline_hybrid
        ),

    "Adaptive Hybrid":
        calculate_metrics(
            gray,
            adaptive_hybrid
        )
}


# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("\nRESULTS")
print("=" * 70)

for method, metrics in results.items():

    print(
        f"{method:<20}"
        f"MSE = {metrics['MSE']:.4f}   "
        f"PSNR = {metrics['PSNR']:.4f} dB   "
        f"SSIM = {metrics['SSIM']:.4f}"
    )


# ------------------------------------------------------------
# Save images
# ------------------------------------------------------------

cv2.imwrite(
    "results/images/knee_he.png",
    he
)

cv2.imwrite(
    "results/images/knee_clahe.png",
    clahe
)

cv2.imwrite(
    "results/images/knee_baseline_hybrid.png",
    baseline_hybrid
)

cv2.imwrite(
    "results/images/knee_adaptive_hybrid.png",
    adaptive_hybrid
)

print("\nImages saved in results/images/")