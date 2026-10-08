import numpy as np

from skimage.metrics import structural_similarity


def mse(reference, enhanced):
    """
    Mean Squared Error.

    Lower is better.
    """

    reference = reference.astype(np.float64)
    enhanced = enhanced.astype(np.float64)

    return float(
        np.mean(
            (reference - enhanced) ** 2
        )
    )


def psnr(reference, enhanced):
    """
    Peak Signal-to-Noise Ratio.

    Higher is better.
    Images are assumed to be 8-bit grayscale.
    """

    error = mse(reference, enhanced)

    if error == 0:
        return float("inf")

    max_pixel = 255.0

    return float(
        10 * np.log10(
            (max_pixel ** 2) / error
        )
    )


def ssim(reference, enhanced):
    """
    Structural Similarity Index.

    Higher and closer to 1 is better.
    """

    return float(
        structural_similarity(
            reference,
            enhanced,
            data_range=255
        )
    )


def calculate_metrics(reference, enhanced):
    """
    Calculate all three evaluation metrics.
    """

    return {
        "MSE": mse(
            reference,
            enhanced
        ),

        "PSNR": psnr(
            reference,
            enhanced
        ),

        "SSIM": ssim(
            reference,
            enhanced
        )
    }