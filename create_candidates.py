import h5py
import numpy as np
import cv2
import os


# ============================================================
# SETTINGS
# ============================================================

DATASETS = {
    "xray": "data/mendeley/test/chestxray14_test.h5",
    "ct": "data/mendeley/test/lidcidri2d_test.h5",
    "mri": "data/mendeley/test/openbhb2d_test.h5",
}

OUTPUT_DIR = "data/candidates"

# Number of candidate images to create
NUM_CANDIDATES = 20


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def is_good_image(image):
    """
    Reject images that are almost completely black,
    almost completely white, or have very little variation.
    """

    mean = np.mean(image)
    std = np.std(image)

    # Reject nearly black images
    if mean < 10:
        return False

    # Reject nearly white images
    if mean > 245:
        return False

    # Reject images with almost no contrast
    if std < 15:
        return False

    return True


# ============================================================
# PROCESS EACH MODALITY
# ============================================================

for modality, file_path in DATASETS.items():

    print("\n" + "=" * 70)
    print(f"Processing {modality.upper()}")
    print("=" * 70)

    output_folder = os.path.join(
        OUTPUT_DIR,
        modality
    )

    os.makedirs(output_folder, exist_ok=True)

    with h5py.File(file_path, "r") as f:

        images = f["images"]

        total = len(images)

        # Spread candidate indices across the dataset
        indices = np.linspace(
            0,
            total - 1,
            NUM_CANDIDATES * 5,
            dtype=int
        )

        selected = []

        for index in indices:

            image = images[index]

            if is_good_image(image):

                selected.append(index)

            if len(selected) >= NUM_CANDIDATES:
                break

        print(f"Total images: {total}")
        print(f"Selected candidates: {len(selected)}")

        # Save candidate images
        for number, index in enumerate(selected, start=1):

            image = images[index]

            image = np.asarray(
                image,
                dtype=np.uint8
            )

            # Dataset is already grayscale.
            # Resize to paper's 255 × 255 format.
            image = cv2.resize(
                image,
                (255, 255),
                interpolation=cv2.INTER_AREA
            )

            filename = (
                f"{modality}_candidate_{number:02d}"
                f"_idx_{index}.png"
            )

            output_path = os.path.join(
                output_folder,
                filename
            )

            cv2.imwrite(
                output_path,
                image
            )

            print(
                f"Saved: {output_path}"
            )

print("\nCandidate extraction complete.")