import cv2
import matplotlib.pyplot as plt


def create_comparison_figure(
    original,
    he_image,
    clahe_image,
    hybrid_image,
    adaptive_hybrid
):

    images = [
        original,
        he_image,
        clahe_image,
        hybrid_image,
        adaptive_hybrid
    ]

    titles = [
        "Original",
        "HE",
        "CLAHE",
        "Hybrid HE→CLAHE",
        "Adaptive Hybrid"
    ]

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(18, 7)
    )

    for i, (image, title) in enumerate(
        zip(images, titles)
    ):

        # Image
        axes[0, i].imshow(
            image,
            cmap="gray"
        )

        axes[0, i].set_title(
            title
        )

        axes[0, i].axis("off")

        # Histogram
        axes[1, i].hist(
            image.ravel(),
            bins=256,
            range=(0, 256)
        )

        axes[1, i].set_title(
            f"{title} Histogram"
        )

        axes[1, i].set_xlabel(
            "Pixel Intensity"
        )

        axes[1, i].set_ylabel(
            "Frequency"
        )

    plt.tight_layout()

    return fig


def create_metric_figure(results):
    """
    Create MSE, PSNR and SSIM comparison charts.
    """

    methods = list(results.keys())

    mse_values = [
        results[m]["MSE"]
        for m in methods
    ]

    psnr_values = [
        results[m]["PSNR"]
        for m in methods
    ]

    ssim_values = [
        results[m]["SSIM"]
        for m in methods
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # MSE
    axes[0].bar(
        methods,
        mse_values
    )

    axes[0].set_title(
        "MSE Comparison"
    )

    axes[0].set_ylabel(
        "MSE"
    )

    axes[0].tick_params(
        axis="x",
        rotation=20
    )

    # PSNR
    axes[1].bar(
        methods,
        psnr_values
    )

    axes[1].set_title(
        "PSNR Comparison"
    )

    axes[1].set_ylabel(
        "PSNR (dB)"
    )

    axes[1].tick_params(
        axis="x",
        rotation=20
    )

    # SSIM
    axes[2].bar(
        methods,
        ssim_values
    )

    axes[2].set_title(
        "SSIM Comparison"
    )

    axes[2].set_ylabel(
        "SSIM"
    )

    axes[2].set_ylim(
        0,
        1
    )

    axes[2].tick_params(
        axis="x",
        rotation=20
    )

    plt.tight_layout()

    return fig