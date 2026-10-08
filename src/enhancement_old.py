import cv2
import numpy as np


# ============================================================
# 1. GLOBAL HISTOGRAM EQUALIZATION
# ============================================================

def histogram_equalization(image):
    """
    Global Histogram Equalization (HE).

    HE works on the complete image histogram and redistributes
    pixel intensities across the available 0-255 range.

    Input:
        image : 8-bit grayscale image

    Returns:
        Enhanced 8-bit grayscale image
    """

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    return cv2.equalizeHist(image)


# ============================================================
# 2. STANDARD CLAHE
# ============================================================

def clahe_enhancement(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Contrast-Limited Adaptive Histogram Equalization (CLAHE).

    Parameters:
        image:
            8-bit grayscale image.

        clip_limit:
            Controls contrast amplification.

        tile_grid_size:
            Number of local tiles.

            (8, 8) means:
                8 x 8 = 64 local regions.

    Returns:
        CLAHE enhanced image.
    """

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(image)


# ============================================================
# 3. BASELINE HYBRID HE + CLAHE
# ============================================================

def hybrid_he_clahe(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Baseline sequential hybrid.

    Processing:

        Original
            ↓
           HE
            ↓
          CLAHE
            ↓
         Output

    This provides a simple baseline against which the
    adaptive hybrid can be compared.
    """

    he_image = histogram_equalization(
        image
    )

    hybrid = clahe_enhancement(
        he_image,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size
    )

    return hybrid


# ============================================================
# 4. HISTOGRAM CLIPPING
# ============================================================

def clip_histogram(
    histogram,
    clip_threshold
):
    """
    Clip histogram bins above a specified threshold.

    Pixels removed by clipping are redistributed uniformly
    among all histogram bins.

    This implements the basic contrast-limiting idea used
    by CLAHE.
    """

    histogram = histogram.astype(
        np.float64
    )

    # --------------------------------------------------------
    # Find excess pixels
    # --------------------------------------------------------

    excess = np.sum(
        np.maximum(
            histogram - clip_threshold,
            0
        )
    )

    # --------------------------------------------------------
    # Clip histogram
    # --------------------------------------------------------

    clipped = np.minimum(
        histogram,
        clip_threshold
    )

    # --------------------------------------------------------
    # Redistribute excess pixels
    # --------------------------------------------------------

    redistribution = (
        excess / len(clipped)
    )

    clipped += redistribution

    return clipped


# ============================================================
# 5. CALCULATE LOCAL VARIANCE
# ============================================================

def calculate_local_variance(tile):
    """
    Calculate the variance of a local image tile.

    Variance represents the amount of intensity variation
    present in the region.

    Low variance:
        Relatively flat / low-detail region.

    High variance:
        More texture/detail/variation.
    """

    tile_float = tile.astype(
        np.float32
    )

    return float(
        np.var(tile_float)
    )


# ============================================================
# 6. ADAPTIVE CLIP LIMIT
# ============================================================

def calculate_adaptive_clip_limit(
    tile,
    min_clip=1.5,
    max_clip=4.0
):
    """
    Calculate an adaptive clip factor from local variance.

    The research paper states that the hybrid method uses
    local variance to modulate the CLAHE clip limit.

    The paper does NOT provide the exact mathematical
    variance-to-clip-limit mapping.

    Therefore, this function implements an explicit
    normalized-variance mapping:

        Low variance
            ↓
        Higher clip limit
            ↓
        Stronger local enhancement

        High variance
            ↓
        Lower clip limit
            ↓
        More conservative enhancement

    Parameters:
        tile:
            Local image tile.

        min_clip:
            Minimum adaptive clip factor.

        max_clip:
            Maximum adaptive clip factor.

    Returns:
        Adaptive clip factor.
    """

    variance = calculate_local_variance(
        tile
    )

    # Maximum possible variance of an 8-bit
    # image is approximately 255^2 / 4.
    max_possible_variance = (
        255.0 ** 2
    ) / 4.0

    # Normalize variance to [0, 1].
    normalized_variance = (
        variance
        / max_possible_variance
    )

    normalized_variance = np.clip(
        normalized_variance,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Adaptive mapping
    #
    # Low variance  -> max_clip
    # High variance -> min_clip
    # --------------------------------------------------------

    clip_factor = (
        max_clip
        -
        normalized_variance
        *
        (max_clip - min_clip)
    )

    clip_factor = np.clip(
        clip_factor,
        min_clip,
        max_clip
    )

    return float(
        clip_factor
    )


# ============================================================
# 7. CREATE LOCAL LUT
# ============================================================

def create_tile_lut(
    tile,
    clip_factor
):
    """
    Create a histogram-equalization lookup table (LUT)
    for one local tile.

    Processing:

        Tile
          ↓
      Histogram
          ↓
    Clip Histogram
          ↓
        CDF
          ↓
    Normalize CDF
          ↓
        LUT
    """

    # --------------------------------------------------------
    # Calculate histogram
    # --------------------------------------------------------

    histogram, _ = np.histogram(
        tile.flatten(),
        bins=256,
        range=(0, 256)
    )

    histogram = histogram.astype(
        np.float64
    )

    # --------------------------------------------------------
    # Calculate clip threshold
    # --------------------------------------------------------

    tile_height, tile_width = (
        tile.shape
    )

    tile_area = (
        tile_height
        *
        tile_width
    )

    clip_threshold = max(
        1.0,
        clip_factor
        *
        tile_area
        /
        256.0
    )

    # --------------------------------------------------------
    # Clip histogram
    # --------------------------------------------------------

    clipped_histogram = clip_histogram(
        histogram,
        clip_threshold
    )

    # --------------------------------------------------------
    # Calculate CDF
    # --------------------------------------------------------

    cdf = np.cumsum(
        clipped_histogram
    )

    non_zero = cdf[
        cdf > 0
    ]

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(non_zero) == 0:

        return np.arange(
            256,
            dtype=np.uint8
        )

    cdf_min = non_zero[0]

    denominator = (
        cdf[-1]
        -
        cdf_min
    )

    if denominator <= 0:

        return np.arange(
            256,
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # Normalize CDF to 0-255
    # --------------------------------------------------------

    lut = (
        (cdf - cdf_min)
        /
        denominator
        *
        255.0
    )

    lut = np.clip(
        lut,
        0,
        255
    )

    return lut.astype(
        np.uint8
    )


# ============================================================
# 8. ADAPTIVE 64-BLOCK CLAHE
# ============================================================

def adaptive_clahe_64_blocks(
    image,
    min_clip=1.5,
    max_clip=4.0,
    grid_size=8
):
    """
    Paper-inspired adaptive CLAHE.

    Main processing:

        HE image
           ↓
        8 x 8 grid
           ↓
        64 blocks
           ↓
    Local variance
           ↓
    Adaptive clip limit
           ↓
    Local histogram
           ↓
    Histogram clipping
           ↓
          CDF
           ↓
          LUT
           ↓
    Bilinear interpolation
           ↓
        Final image

    Important:
        The paper specifies 64 blocks and a content-aware
        clipping mechanism based on local variance.

        The exact variance-to-clip-limit equation is not
        provided in the paper, so our mapping is explicitly
        implemented in calculate_adaptive_clip_limit().
    """

    if image.dtype != np.uint8:
        image = image.astype(
            np.uint8
        )

    height, width = image.shape

    # ========================================================
    # STEP 1
    # Divide image into grid
    # ========================================================

    y_boundaries = np.linspace(
        0,
        height,
        grid_size + 1,
        dtype=int
    )

    x_boundaries = np.linspace(
        0,
        width,
        grid_size + 1,
        dtype=int
    )

    # ========================================================
    # STEP 2
    # Create LUT for every tile
    # ========================================================

    tile_luts = []

    for row in range(grid_size):

        row_luts = []

        y1 = y_boundaries[row]
        y2 = y_boundaries[row + 1]

        for col in range(grid_size):

            x1 = x_boundaries[col]
            x2 = x_boundaries[col + 1]

            # ------------------------------------------------
            # Extract tile
            # ------------------------------------------------

            tile = image[
                y1:y2,
                x1:x2
            ]

            # ------------------------------------------------
            # Calculate adaptive clip factor
            # ------------------------------------------------

            clip_factor = (
                calculate_adaptive_clip_limit(
                    tile,
                    min_clip=min_clip,
                    max_clip=max_clip
                )
            )

            # ------------------------------------------------
            # Generate LUT
            # ------------------------------------------------

            lut = create_tile_lut(
                tile,
                clip_factor
            )

            row_luts.append(
                lut
            )

        tile_luts.append(
            row_luts
        )

    # ========================================================
    # STEP 3
    # Calculate tile centers
    # ========================================================

    y_centers = np.array([
        (
            y_boundaries[i]
            +
            y_boundaries[i + 1]
            -
            1
        ) / 2.0

        for i in range(grid_size)
    ])

    x_centers = np.array([
        (
            x_boundaries[i]
            +
            x_boundaries[i + 1]
            -
            1
        ) / 2.0

        for i in range(grid_size)
    ])

    # ========================================================
    # STEP 4
    # Prepare output
    # ========================================================

    output = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # ========================================================
    # STEP 5
    # Bilinear interpolation
    # ========================================================

    for y in range(height):

        # ----------------------------------------------------
        # Find vertical neighbouring tiles
        # ----------------------------------------------------

        if y <= y_centers[0]:

            row0 = 0
            row1 = 0
            beta = 0.0

        elif y >= y_centers[-1]:

            row0 = grid_size - 1
            row1 = grid_size - 1
            beta = 0.0

        else:

            row1 = np.searchsorted(
                y_centers,
                y
            )

            row0 = row1 - 1

            beta = (
                y
                -
                y_centers[row0]
            ) / (
                y_centers[row1]
                -
                y_centers[row0]
            )

        for x in range(width):

            # ------------------------------------------------
            # Find horizontal neighbouring tiles
            # ------------------------------------------------

            if x <= x_centers[0]:

                col0 = 0
                col1 = 0
                alpha = 0.0

            elif x >= x_centers[-1]:

                col0 = grid_size - 1
                col1 = grid_size - 1
                alpha = 0.0

            else:

                col1 = np.searchsorted(
                    x_centers,
                    x
                )

                col0 = col1 - 1

                alpha = (
                    x
                    -
                    x_centers[col0]
                ) / (
                    x_centers[col1]
                    -
                    x_centers[col0]
                )

            # ------------------------------------------------
            # Original intensity of current pixel
            # ------------------------------------------------

            intensity = int(
                image[y, x]
            )

            # ------------------------------------------------
            # Apply each neighbouring LUT to the SAME
            # original intensity value.
            # ------------------------------------------------

            value00 = tile_luts[
                row0
            ][
                col0
            ][
                intensity
            ]

            value01 = tile_luts[
                row0
            ][
                col1
            ][
                intensity
            ]

            value10 = tile_luts[
                row1
            ][
                col0
            ][
                intensity
            ]

            value11 = tile_luts[
                row1
            ][
                col1
            ][
                intensity
            ]

            # ------------------------------------------------
            # Horizontal interpolation - top
            # ------------------------------------------------

            top_value = (
                (1.0 - alpha)
                *
                value00
                +
                alpha
                *
                value01
            )

            # ------------------------------------------------
            # Horizontal interpolation - bottom
            # ------------------------------------------------

            bottom_value = (
                (1.0 - alpha)
                *
                value10
                +
                alpha
                *
                value11
            )

            # ------------------------------------------------
            # Vertical interpolation
            # ------------------------------------------------

            final_value = (
                (1.0 - beta)
                *
                top_value
                +
                beta
                *
                bottom_value
            )

            output[y, x] = (
                final_value
            )

    # ========================================================
    # STEP 6
    # Convert output to valid 8-bit image
    # ========================================================

    output = np.clip(
        output,
        0,
        255
    )

    return output.astype(
        np.uint8
    )


# ============================================================
# 9. MAIN PAPER-INSPIRED HYBRID
# ============================================================

def adaptive_hybrid_he_clahe(
    image,
    min_clip=1.5,
    max_clip=4.0,
    grid_size=8
):
    """
    Main paper-inspired Hybrid HE-CLAHE method.

    Processing:

        Original
            ↓
        Histogram Equalization
            ↓
        8 x 8 = 64 blocks
            ↓
        Local variance
            ↓
        Adaptive clip limit
            ↓
        Local histogram equalization
            ↓
        Bilinear interpolation
            ↓
        Hybrid enhanced image
    """

    # --------------------------------------------------------
    # Stage 1: Global Histogram Equalization
    # --------------------------------------------------------

    he_image = histogram_equalization(
        image
    )

    # --------------------------------------------------------
    # Stage 2: Adaptive local processing
    # --------------------------------------------------------

    hybrid = adaptive_clahe_64_blocks(
        he_image,
        min_clip=min_clip,
        max_clip=max_clip,
        grid_size=grid_size
    )

    return hybrid