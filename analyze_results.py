import pandas as pd


# ============================================================
# LOAD RESULTS
# ============================================================

CSV_FILE = "results_v3_mendeley/metrics_paper_inspired_v2.csv"

df = pd.read_csv(CSV_FILE)

print("\n")
print("=" * 70)
print("FINAL MEDICAL IMAGE ENHANCEMENT RESULTS")
print("=" * 70)


# ============================================================
# SHOW RAW RESULTS
# ============================================================

print("\nIndividual Image Results")
print("-" * 70)

print(df.to_string(index=False))


# ============================================================
# OVERALL AVERAGE
# ============================================================

overall = (
    df.groupby("Method")[["MSE", "PSNR", "SSIM"]]
    .mean()
    .sort_values("PSNR", ascending=False)
)

print("\n")
print("=" * 70)
print("OVERALL AVERAGE RESULTS")
print("=" * 70)

print(
    overall.round(4).to_string()
)


# ============================================================
# MODALITY-WISE RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("MODALITY-WISE RESULTS")
print("=" * 70)

for modality in ["X-Ray", "CT", "MRI"]:

    modality_df = df[
        df["Modality"] == modality
    ]

    if modality_df.empty:
        continue

    result = (
        modality_df
        .groupby("Method")[["MSE", "PSNR", "SSIM"]]
        .mean()
        .sort_values("PSNR", ascending=False)
    )

    print(f"\n{modality}")
    print("-" * 50)

    print(
        result.round(4).to_string()
    )


# ============================================================
# BEST METHODS
# ============================================================

print("\n")
print("=" * 70)
print("BEST METHODS")
print("=" * 70)


average_results = (
    df.groupby("Method")[["MSE", "PSNR", "SSIM"]]
    .mean()
)


best_mse = average_results["MSE"].idxmin()

best_psnr = average_results["PSNR"].idxmax()

best_ssim = average_results["SSIM"].idxmax()


print(
    f"\nLowest Average MSE  : {best_mse}"
)

print(
    f"Highest Average PSNR: {best_psnr}"
)

print(
    f"Highest Average SSIM: {best_ssim}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = average_results.round(4)

summary.to_csv(
    "results_v3_mendeley/final_average_results.csv"
)

print("\n")
print(
    "Saved: results_v3_mendeley/final_average_results.csv"
)

print("=" * 70)