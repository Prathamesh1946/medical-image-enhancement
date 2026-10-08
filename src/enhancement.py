import cv2
import numpy as np


# ============================================================
# 1. GLOBAL HISTOGRAM EQUALIZATION
# ============================================================

def histogram_equalization(image):
    """
    Global Histogram Equalization (HE).

    The entire image is processed using one global histogram.
    """

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
    Standard Contrast-Limited Adaptive Histogram Equalization.

    This uses OpenCV's CLAHE implementation.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(image)


# ============================================================
# 3. SIMPLE BASELINE HYBRID
# ============================================================

def hybrid_he_clahe(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Baseline sequential hybrid:

        Original
           ↓
          HE
           ↓
         CLAHE

    This is used as a comparison baseline.
    """

    # Step 1: Global HE
    he_image = histogram_equalization(image)

    # Step 2: CLAHE on HE output
    enhanced = clahe_enhancement(
        he_image,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size
    )

    return enhanced


# ============================================================
# 4. HISTOGRAM CLIPPING
# ============================================================

def clip_histogram(histogram, clip_threshold):
    """
    Clip a histogram at a specified threshold and
    redistribute the excess pixels uniformly.

    This is the basic contrast-limiting operation
    used in CLAHE-like processing.
    """

    histogram = histogram.astype(np.float64)

    # --------------------------------------------------------
    # Find pixels above clipping threshold
    # --------------------------------------------------------

    excess = np.maximum(
        histogram - clip_threshold,
        0
    )

    # Remove excess from bins
    clipped_histogram = np.minimum(
        histogram,
        clip_threshold
    )

    total_excess = np.sum(excess)

    # --------------------------------------------------------
    # Redistribute excess uniformly
    # --------------------------------------------------------

    if total_excess > 0:

        clipped_histogram += (
            total_excess / 256.0
        )

    return clipped_histogram


# ============================================================
# 5. CREATE LOCAL LUT
# ============================================================

def create_local_lut(
    block,
    clip_threshold
):
    """
    Create a lookup table for one local image block.

    Steps:

        Histogram
           ↓
        Clip histogram
           ↓
        Redistribute excess
           ↓
        CDF
           ↓
        Normalize CDF
           ↓
        LUT
    """

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    histogram = np.bincount(
        block.ravel(),
        minlength=256
    )

    # --------------------------------------------------------
    # Clip histogram
    # --------------------------------------------------------

    clipped_histogram = clip_histogram(
        histogram,
        clip_threshold
    )

    # --------------------------------------------------------
    # Cumulative Distribution Function
    # --------------------------------------------------------

    cdf = np.cumsum(
        clipped_histogram
    )

    # --------------------------------------------------------
    # Normalize CDF to 0-255
    # --------------------------------------------------------

    cdf_min = cdf[
        np.nonzero(cdf)
    ][0]

    denominator = (
        cdf[-1] - cdf_min
    )

    if denominator <= 0:

        return np.arange(
            256,
            dtype=np.uint8
        )

    lut = (
        (cdf - cdf_min)
        / denominator
        * 255.0
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
# 6. PAPER-INSPIRED ADAPTIVE HYBRID
# ============================================================

def adaptive_hybrid_he_clahe(
    image,
    min_clip=1.0,
    max_clip=2.0,
    grid_size=8
):
    """
    Paper-inspired adaptive HE-CLAHE hybrid.

    Pipeline:

        Original
            ↓
        Global HE
            ↓
        Divide into 8 × 8 = 64 blocks
            ↓
        Calculate local variance
            ↓
        Determine adaptive clipping threshold
            ↓
        Local histogram clipping
            ↓
        Local CDF / LUT
            ↓
        Bilinear interpolation
            ↓
        Final enhanced image

    IMPORTANT:
    The research paper describes the content-aware clipping
    concept but does not publish the exact mathematical
    mapping from local variance to clipping threshold.

    Therefore this is a paper-inspired implementation,
    not a claim of exact source-code reproduction.
    """

    # ========================================================
    # STEP 1: GLOBAL HISTOGRAM EQUALIZATION
    # ========================================================

    he_image = histogram_equalization(
        image
    )

    height, width = he_image.shape

    # ========================================================
    # STEP 2: DETERMINE BLOCK BOUNDARIES
    # ========================================================

    y_edges = np.linspace(
        0,
        height,
        grid_size + 1,
        dtype=int
    )

    x_edges = np.linspace(
        0,
        width,
        grid_size + 1,
        dtype=int
    )

    # ========================================================
    # STEP 3: CALCULATE LOCAL VARIANCES
    # ========================================================

    variances = np.zeros(
        (grid_size, grid_size),
        dtype=np.float64
    )

    blocks = []

    for row in range(grid_size):

        row_blocks = []

        y0 = y_edges[row]
        y1 = y_edges[row + 1]

        for col in range(grid_size):

            x0 = x_edges[col]
            x1 = x_edges[col + 1]

            block = he_image[
                y0:y1,
                x0:x1
            ]

            row_blocks.append(
                block
            )

            variances[
                row,
                col
            ] = np.var(
                block.astype(
                    np.float64
                )
            )

        blocks.append(
            row_blocks
        )

    # ========================================================
    # STEP 4: NORMALIZE LOCAL VARIANCE
    # ========================================================

    minimum_variance = np.min(
        variances
    )

    maximum_variance = np.max(
        variances
    )

    variance_range = (
        maximum_variance
        - minimum_variance
    )

    if variance_range < 1e-12:

        normalized_variance = np.zeros_like(
            variances
        )

    else:

        normalized_variance = (
            variances
            - minimum_variance
        ) / variance_range

    # ========================================================
    # STEP 5: CONTENT-AWARE CLIP LIMIT
    # ========================================================
    #
    # Low variance:
    #     flatter region
    #     → stronger enhancement
    #
    # High variance:
    #     detailed region
    #     → weaker enhancement
    #
    # This follows the paper's qualitative description.
    # ========================================================

    adaptive_clip_limits = (
        max_clip
        - normalized_variance
        * (
            max_clip
            - min_clip
        )
    )

    # ========================================================
    # STEP 6: CREATE LOCAL LUTS
    # ========================================================

    local_luts = np.zeros(
        (
            grid_size,
            grid_size,
            256
        ),
        dtype=np.uint8
    )

    for row in range(grid_size):

        for col in range(grid_size):

            block = blocks[row][col]

            # ------------------------------------------------
            # CLAHE-style clip threshold
            #
            # OpenCV's clipLimit is a normalized factor.
            # We convert it to an actual histogram count
            # using the block's average histogram count.
            # ------------------------------------------------

            pixels_in_block = block.size

            average_bin_count = (
                pixels_in_block
                / 256.0
            )

            clip_threshold = (
                adaptive_clip_limits[row, col]
                * average_bin_count
            )

            clip_threshold = max(
                clip_threshold,
                1.0
            )

            # ------------------------------------------------
            # Generate LUT
            # ------------------------------------------------

            local_luts[
                row,
                col
            ] = create_local_lut(
                block,
                clip_threshold
            )

    # ========================================================
    # STEP 7: BILINEAR INTERPOLATION
    # ========================================================
    #
    # The paper specifies interpolation between blocks
    # to reduce block artifacts.
    # ========================================================

    # Centers of blocks
    y_centers = (
        y_edges[:-1]
        + y_edges[1:]
        - 1
    ) / 2.0

    x_centers = (
        x_edges[:-1]
        + x_edges[1:]
        - 1
    ) / 2.0

    # Output image
    output = np.zeros_like(
        he_image,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Process each row
    # --------------------------------------------------------

    for y in range(height):

        # ----------------------------------------------------
        # Find vertical neighboring blocks
        # ----------------------------------------------------

        if y <= y_centers[0]:

            y0_index = 0
            y1_index = 0
            wy = 0.0

        elif y >= y_centers[-1]:

            y0_index = grid_size - 1
            y1_index = grid_size - 1
            wy = 0.0

        else:

            y1_index = np.searchsorted(
                y_centers,
                y
            )

            y0_index = y1_index - 1

            distance = (
                y_centers[y1_index]
                - y_centers[y0_index]
            )

            if distance == 0:

                wy = 0.0

            else:

                wy = (
                    y
                    - y_centers[y0_index]
                ) / distance

        # ----------------------------------------------------
        # Process columns
        # ----------------------------------------------------

        for x in range(width):

            # -----------------------------------------------
            # Find horizontal neighboring blocks
            # -----------------------------------------------

            if x <= x_centers[0]:

                x0_index = 0
                x1_index = 0
                wx = 0.0

            elif x >= x_centers[-1]:

                x0_index = grid_size - 1
                x1_index = grid_size - 1
                wx = 0.0

            else:

                x1_index = np.searchsorted(
                    x_centers,
                    x
                )

                x0_index = x1_index - 1

                distance = (
                    x_centers[x1_index]
                    - x_centers[x0_index]
                )

                if distance == 0:

                    wx = 0.0

                else:

                    wx = (
                        x
                        - x_centers[x0_index]
                    ) / distance

            # -----------------------------------------------
            # Pixel intensity from HE image
            # -----------------------------------------------

            intensity = int(
                he_image[y, x]
            )

            # -----------------------------------------------
            # Four neighboring LUT outputs
            # -----------------------------------------------

            top_left = float(
                local_luts[
                    y0_index,
                    x0_index,
                    intensity
                ]
            )

            top_right = float(
                local_luts[
                    y0_index,
                    x1_index,
                    intensity
                ]
            )

            bottom_left = float(
                local_luts[
                    y1_index,
                    x0_index,
                    intensity
                ]
            )

            bottom_right = float(
                local_luts[
                    y1_index,
                    x1_index,
                    intensity
                ]
            )

            # -----------------------------------------------
            # Horizontal interpolation
            # -----------------------------------------------

            top = (
                (1.0 - wx)
                * top_left
                +
                wx
                * top_right
            )

            bottom = (
                (1.0 - wx)
                * bottom_left
                +
                wx
                * bottom_right
            )

            # -----------------------------------------------
            # Vertical interpolation
            # -----------------------------------------------

            value = (
                (1.0 - wy)
                * top
                +
                wy
                * bottom
            )

            output[y, x] = value

    # ========================================================
    # STEP 8: CONVERT TO UINT8
    # ========================================================

    output = np.clip(
        output,
        0,
        255
    ).astype(
        np.uint8
    )

    return output