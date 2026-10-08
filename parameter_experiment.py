import cv2
import pandas as pd
import os

from src.preprocessing import preprocess_image
from src.enhancement import adaptive_hybrid_he_clahe
from src.metrics import calculate_metrics


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = "dataset/xray/knee.png"

OUTPUT_DIR = "results/parameter_experiment"


# ============================================================
# PARAMETER COMBINATIONS
# ============================================================

PARAMETERS = [
    (1.0, 2.0),
    (1.0, 2.5),
    (1.0, 3.0),
    (1.0, 3.5),

    (1.5, 2.0),
    (1.5, 2.5),
    (1.5, 3.0),
    (1.5, 3.5),
    (1.5, 4.0),

    (2.0, 2.5),
    (2.0, 3.0),
    (2.0, 3.5),
    (2.0, 4.0),
]


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading image...")

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Could not find image: {IMAGE_PATH}"
    )


# ============================================================
# PREPROCESS
# ============================================================

print("Preprocessing image...")

original = preprocess_image(
    image
)

print(
    f"Image size: {original.shape}"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []


print()
print("=" * 75)
print("ADAPTIVE CLIP-LIMIT PARAMETER EXPERIMENT")
print("=" * 75)


for min_clip, max_clip in PARAMETERS:

    print(
        f"Testing Min Clip = {min_clip}, "
        f"Max Clip = {max_clip}"
    )

    # --------------------------------------------------------
    # Run adaptive hybrid
    # --------------------------------------------------------

    enhanced = adaptive_hybrid_he_clahe(
        original,
        min_clip=min_clip,
        max_clip=max_clip,
        grid_size=8
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        original,
        enhanced
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({

        "Min Clip":
            min_clip,

        "Max Clip":
            max_clip,

        "MSE":
            metrics["MSE"],

        "PSNR":
            metrics["PSNR"],

        "SSIM":
            metrics["SSIM"]
    })

    # --------------------------------------------------------
    # Save enhanced image
    # --------------------------------------------------------

    filename = (
        f"adaptive_"
        f"min{str(min_clip).replace('.', '_')}_"
        f"max{str(max_clip).replace('.', '_')}.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(
        output_path,
        enhanced
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# SORT RESULTS
# ============================================================

df_sorted = df.sort_values(
    by="PSNR",
    ascending=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 75)
print("RESULTS SORTED BY PSNR")
print("=" * 75)

print(
    df_sorted.to_string(
        index=False
    )
)


# ============================================================
# BEST RESULTS
# ============================================================

best_psnr = df.loc[
    df["PSNR"].idxmax()
]

best_ssim = df.loc[
    df["SSIM"].idxmax()
]

best_mse = df.loc[
    df["MSE"].idxmin()
]


print()
print("=" * 75)
print("BEST CONFIGURATIONS")
print("=" * 75)


print(
    "\nBest PSNR:"
)

print(
    f"Min Clip = {best_psnr['Min Clip']}"
)

print(
    f"Max Clip = {best_psnr['Max Clip']}"
)

print(
    f"PSNR = {best_psnr['PSNR']:.4f} dB"
)


print(
    "\nBest SSIM:"
)

print(
    f"Min Clip = {best_ssim['Min Clip']}"
)

print(
    f"Max Clip = {best_ssim['Max Clip']}"
)

print(
    f"SSIM = {best_ssim['SSIM']:.4f}"
)


print(
    "\nLowest MSE:"
)

print(
    f"Min Clip = {best_mse['Min Clip']}"
)

print(
    f"Max Clip = {best_mse['Max Clip']}"
)

print(
    f"MSE = {best_mse['MSE']:.4f}"
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "parameter_results.csv"
)

df.to_csv(
    csv_path,
    index=False
)


print()
print(
    f"CSV saved to: {csv_path}"
)

print(
    f"Images saved to: {OUTPUT_DIR}"
)