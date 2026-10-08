import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import (
    load_image,
    preprocess_image
)

from src.enhancement import (
    histogram_equalization,
    clahe_enhancement,
    hybrid_he_clahe,
    adaptive_hybrid_he_clahe
)

from src.metrics import (
    calculate_metrics
)

from src.visualization import (
    create_comparison_figure,
    create_metric_figure
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Image Enhancement",
    page_icon="🩻",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "Medical Image Enhancement"
)

st.subheader(
    "Hybrid Histogram Equalization – CLAHE"
)

st.write(
    """
    A research-paper-inspired medical image enhancement
    system using Histogram Equalization (HE),
    Contrast-Limited Adaptive Histogram Equalization
    (CLAHE), a sequential HE→CLAHE baseline,
    and an adaptive hybrid enhancement technique.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Parameters"
)

clip_limit = st.sidebar.slider(
    "CLAHE Clip Limit",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5
)

grid_size = st.sidebar.selectbox(
    "Grid Size",
    [4, 8, 16],
    index=1
)

min_clip = st.sidebar.slider(
    "Adaptive Minimum Clip",
    1.0,
    3.0,
    1.5,
    0.1
)

max_clip = st.sidebar.slider(
    "Adaptive Maximum Clip",
    2.0,
    6.0,
    4.0,
    0.1
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff"
    ]
)


# ============================================================
# MAIN PIPELINE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Load Image
    # --------------------------------------------------------

    input_image = load_image(
        uploaded_file
    )

    # --------------------------------------------------------
    # Preprocess Image
    # --------------------------------------------------------

    original = preprocess_image(
        input_image
    )

    st.success(
        "Image successfully loaded and preprocessed."
    )

    st.write(
        f"Processed resolution: "
        f"{original.shape[1]} × {original.shape[0]}"
    )

    # --------------------------------------------------------
    # Apply Algorithms
    # --------------------------------------------------------

    # 1. Histogram Equalization
    he_image = histogram_equalization(
        original
    )

    # 2. CLAHE
    clahe_image = clahe_enhancement(
        original,
        clip_limit=clip_limit,
        tile_grid_size=(
            grid_size,
            grid_size
        )
    )

    # 3. Sequential Hybrid HE → CLAHE
    baseline_hybrid = hybrid_he_clahe(
        original,
        clip_limit=clip_limit,
        tile_grid_size=(
            grid_size,
            grid_size
        )
    )

    # 4. Adaptive Hybrid
    adaptive_hybrid = adaptive_hybrid_he_clahe(
        original,
        min_clip=min_clip,
        max_clip=max_clip,
        grid_size=grid_size
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    results = {

        "HE":
            calculate_metrics(
                original,
                he_image
            ),

        "CLAHE":
            calculate_metrics(
                original,
                clahe_image
            ),

        "Hybrid HE→CLAHE":
            calculate_metrics(
                original,
                baseline_hybrid
            ),

        "Adaptive Hybrid":
            calculate_metrics(
                original,
                adaptive_hybrid
            )
    }

    # ========================================================
    # ENHANCED IMAGES
    # ========================================================

    st.header(
        "Enhanced Images"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    with col1:

        st.image(
            original,
            caption="Original",
            clamp=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # HE
    # --------------------------------------------------------

    with col2:

        st.image(
            he_image,
            caption="HE",
            clamp=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    with col3:

        st.image(
            clahe_image,
            caption="CLAHE",
            clamp=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Sequential Hybrid
    # --------------------------------------------------------

    with col4:

        st.image(
            baseline_hybrid,
            caption="Hybrid HE→CLAHE",
            clamp=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Adaptive Hybrid
    # --------------------------------------------------------

    with col5:

        st.image(
            adaptive_hybrid,
            caption="Adaptive Hybrid",
            clamp=True,
            use_container_width=True
        )

    # ========================================================
    # PERFORMANCE COMPARISON
    # ========================================================

    st.header(
        "Performance Comparison"
    )

    metric_rows = []

    for method, values in results.items():

        metric_rows.append({

            "Method":
                method,

            "MSE":
                round(
                    values["MSE"],
                    4
                ),

            "PSNR (dB)":
                round(
                    values["PSNR"],
                    4
                ),

            "SSIM":
                round(
                    values["SSIM"],
                    4
                )
        })

    metrics_df = pd.DataFrame(
        metric_rows
    )

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # BEST METHOD
    # ========================================================

    best_psnr_method = max(
        results,
        key=lambda method:
            results[method]["PSNR"]
    )

    best_ssim_method = max(
        results,
        key=lambda method:
            results[method]["SSIM"]
    )

    best_mse_method = min(
        results,
        key=lambda method:
            results[method]["MSE"]
    )

    st.success(
        f"""
        Best PSNR: {best_psnr_method}

        Best SSIM: {best_ssim_method}

        Lowest MSE: {best_mse_method}
        """
    )

    # ========================================================
    # IMAGE + HISTOGRAM ANALYSIS
    # ========================================================

    st.header(
        "Image and Histogram Analysis"
    )

    figure = create_comparison_figure(
        original,
        he_image,
        clahe_image,
        baseline_hybrid,
        adaptive_hybrid
    )

    st.pyplot(
        figure,
        clear_figure=True
    )

    # ========================================================
    # METRIC GRAPHS
    # ========================================================

    st.header(
        "Metric Graphs"
    )

    metric_figure = create_metric_figure(
        results
    )

    st.pyplot(
        metric_figure,
        clear_figure=True
    )


# ============================================================
# NO IMAGE UPLOADED
# ============================================================

else:

    st.info(
        "Upload an X-ray, CT or MRI image to begin."
    )

    st.markdown(
        """
        ### Processing Pipeline

        **Input Image**
        ↓

        **Resize to 255 × 255**
        ↓

        **Grayscale Conversion**
        ↓

        **HE / CLAHE / Hybrid HE→CLAHE / Adaptive Hybrid**
        ↓

        **MSE / PSNR / SSIM**
        ↓

        **Visual and Quantitative Comparison**
        """
    )