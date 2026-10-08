import os
import cv2
import pandas as pd

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

DATASET = {
    "X-Ray": "data/processed/xray",
    "CT": "data/processed/ct",
    "MRI": "data/processed/mri"
}

RESULTS_DIR = "results_v3_mendeley"


# ============================================================
# ADAPTIVE HYBRID PARAMETERS
# ============================================================

# These were selected from our controlled parameter experiment
# on the knee X-ray.

MIN_CLIP = 1.0
MAX_CLIP = 2.0

# 8 x 8 = 64 blocks
GRID_SIZE = 8


# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    modality,
    filename
):
    """
    Process one medical image using:

        1. Global Histogram Equalization
        2. CLAHE
        3. Baseline Hybrid HE -> CLAHE
        4. Adaptive Hybrid HE -> 64-block adaptive CLAHE

    Then calculate:

        MSE
        PSNR
        SSIM

    Returns:
        List of result dictionaries.
    """

    print()
    print("-" * 70)
    print(
        f"Processing: {modality} / {filename}"
    )
    print("-" * 70)


    # ========================================================
    # STEP 1: LOAD IMAGE
    # ========================================================

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"ERROR: Could not load {image_path}"
        )

        return []


    # ========================================================
    # STEP 2: PREPROCESS
    # ========================================================

    original = preprocess_image(
        image
    )

    print(
        f"Preprocessed size: "
        f"{original.shape[1]} x {original.shape[0]}"
    )


    # ========================================================
    # STEP 3: HISTOGRAM EQUALIZATION
    # ========================================================

    print(
        "Running Histogram Equalization..."
    )

    he_image = histogram_equalization(
        original
    )


    # ========================================================
    # STEP 4: STANDARD CLAHE
    # ========================================================

    print(
        "Running CLAHE..."
    )

    clahe_image = clahe_enhancement(
        original,
        clip_limit=2.0,
        tile_grid_size=(
            8,
            8
        )
    )


    # ========================================================
    # STEP 5: BASELINE HYBRID
    # ========================================================

    print(
        "Running baseline HE -> CLAHE..."
    )

    baseline_hybrid = hybrid_he_clahe(
        original,
        clip_limit=2.0,
        tile_grid_size=(
            8,
            8
        )
    )


    # ========================================================
    # STEP 6: ADAPTIVE HYBRID
    # ========================================================

    print(
        "Running adaptive HE -> CLAHE..."
    )

    adaptive_hybrid = adaptive_hybrid_he_clahe(
        original,
        min_clip=MIN_CLIP,
        max_clip=MAX_CLIP,
        grid_size=GRID_SIZE
    )


    # ========================================================
    # STEP 7: STORE ENHANCED IMAGES
    # ========================================================

    enhanced_images = {

        "HE":
            he_image,

        "CLAHE":
            clahe_image,

        "Hybrid HE-CLAHE":
            baseline_hybrid,

        "Adaptive Hybrid":
            adaptive_hybrid
    }


    # ========================================================
    # STEP 8: CALCULATE METRICS
    # ========================================================

    rows = []


    for method, enhanced_image in enhanced_images.items():

        print(
            f"Calculating metrics for {method}..."
        )


        metrics = calculate_metrics(
            original,
            enhanced_image
        )


        rows.append({

            "Modality":
                modality,

            "Image":
                filename,

            "Method":
                method,

            "MSE":
                metrics["MSE"],

            "PSNR":
                metrics["PSNR"],

            "SSIM":
                metrics["SSIM"]
        })


    # ========================================================
    # STEP 9: SAVE ENHANCED IMAGES
    # ========================================================

    # Create a folder for this image.

    image_name = os.path.splitext(
        filename
    )[0]

    image_output_dir = os.path.join(
        RESULTS_DIR,
        "images",
        modality.lower().replace(
            "-",
            "_"
        ),
        image_name
    )


    os.makedirs(
        image_output_dir,
        exist_ok=True
    )


    # Save original image

    cv2.imwrite(
        os.path.join(
            image_output_dir,
            "original.png"
        ),
        original
    )


    # Save each enhanced image

    for method, enhanced_image in enhanced_images.items():

        safe_method_name = (
            method
            .lower()
            .replace(
                " ",
                "_"
            )
            .replace(
                "→",
                "_"
            )
            .replace(
                "-",
                "_"
            )
        )


        output_path = os.path.join(
            image_output_dir,
            f"{safe_method_name}.png"
        )


        cv2.imwrite(
            output_path,
            enhanced_image
        )


    print(
        f"Saved images to: "
        f"{image_output_dir}"
    )


    return rows


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MEDICAL IMAGE ENHANCEMENT")
    print("RESEARCH PAPER EXPERIMENT")
    print("=" * 70)

    print()
    print(
        f"Adaptive Minimum Clip : {MIN_CLIP}"
    )

    print(
        f"Adaptive Maximum Clip : {MAX_CLIP}"
    )

    print(
        f"Grid Size             : "
        f"{GRID_SIZE} x {GRID_SIZE}"
    )

    print(
        f"Total Blocks          : "
        f"{GRID_SIZE * GRID_SIZE}"
    )


    # ========================================================
    # CREATE RESULT DIRECTORIES
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            RESULTS_DIR,
            "images"
        ),
        exist_ok=True
    )


    # ========================================================
    # STORE ALL RESULTS
    # ========================================================

    all_results = []


    # ========================================================
    # PROCESS EACH MODALITY
    # ========================================================

    for modality, folder in DATASET.items():

        print()
        print()
        print("#" * 70)
        print(
            f"MODALITY: {modality}"
        )
        print("#" * 70)


        # ----------------------------------------------------
        # Check folder
        # ----------------------------------------------------

        if not os.path.exists(folder):

            print(
                f"WARNING: Folder does not exist:"
                f" {folder}"
            )

            continue


        # ----------------------------------------------------
        # Get image files
        # ----------------------------------------------------

        files = sorted(
            [
                filename
                for filename in os.listdir(folder)
                if filename.lower().endswith(
                    SUPPORTED_EXTENSIONS
                )
            ]
        )


        if len(files) == 0:

            print(
                f"WARNING: No images found in {folder}"
            )

            continue


        print(
            f"Found {len(files)} image(s)"
        )


        # ----------------------------------------------------
        # Process images
        # ----------------------------------------------------

        for filename in files:

            image_path = os.path.join(
                folder,
                filename
            )


            rows = process_image(
                image_path,
                modality,
                filename
            )


            all_results.extend(
                rows
            )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if len(all_results) == 0:

        print()
        print(
            "ERROR: No images were processed."
        )

        print(
            "Please add images to:"
        )

        print(
            "dataset/xray/"
        )

        print(
            "dataset/ct/"
        )

        print(
            "dataset/mri/"
        )

        return


    dataframe = pd.DataFrame(
        all_results
    )


    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    csv_path = os.path.join(
    RESULTS_DIR,
    "metrics_paper_inspired_v2.csv"
)


    dataframe.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # DISPLAY COMPLETE RESULTS
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("COMPLETE EXPERIMENT RESULTS")
    print("=" * 90)

    print(
        dataframe.to_string(
            index=False
        )
    )


    # ========================================================
    # OVERALL METHOD AVERAGES
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("OVERALL METHOD AVERAGES")
    print("=" * 90)


    overall = (
        dataframe
        .groupby("Method")
        [
            [
                "MSE",
                "PSNR",
                "SSIM"
            ]
        ]
        .mean()
        .sort_values(
            "PSNR",
            ascending=False
        )
    )


    print(
        overall.to_string()
    )


    # ========================================================
    # MODALITY-WISE AVERAGES
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("MODALITY-WISE AVERAGES")
    print("=" * 90)


    modality_average = (
        dataframe
        .groupby(
            [
                "Modality",
                "Method"
            ]
        )
        [
            [
                "MSE",
                "PSNR",
                "SSIM"
            ]
        ]
        .mean()
    )


    print(
        modality_average.to_string()
    )


    # ========================================================
    # BEST METHODS
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("BEST METHODS")
    print("=" * 90)


    # --------------------------------------------------------
    # Best overall MSE
    # --------------------------------------------------------

    best_mse_method = (
        overall["MSE"]
        .idxmin()
    )

    best_mse_value = (
        overall.loc[
            best_mse_method,
            "MSE"
        ]
    )


    # --------------------------------------------------------
    # Best overall PSNR
    # --------------------------------------------------------

    best_psnr_method = (
        overall["PSNR"]
        .idxmax()
    )

    best_psnr_value = (
        overall.loc[
            best_psnr_method,
            "PSNR"
        ]
    )


    # --------------------------------------------------------
    # Best overall SSIM
    # --------------------------------------------------------

    best_ssim_method = (
        overall["SSIM"]
        .idxmax()
    )

    best_ssim_value = (
        overall.loc[
            best_ssim_method,
            "SSIM"
        ]
    )


    print()
    print(
        f"Lowest Average MSE:"
    )

    print(
        f"    {best_mse_method}"
    )

    print(
        f"    MSE = {best_mse_value:.4f}"
    )


    print()
    print(
        f"Highest Average PSNR:"
    )

    print(
        f"    {best_psnr_method}"
    )

    print(
        f"    PSNR = {best_psnr_value:.4f} dB"
    )


    print()
    print(
        f"Highest Average SSIM:"
    )

    print(
        f"    {best_ssim_method}"
    )

    print(
        f"    SSIM = {best_ssim_value:.4f}"
    )


    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 90)
    print("EXPERIMENT COMPLETED")
    print("=" * 90)

    print()
    print(
        f"Metrics CSV:"
    )

    print(
        f"    {csv_path}"
    )

    print()
    print(
        "Enhanced images:"
    )

    print(
        f"    {RESULTS_DIR}/images/"
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()