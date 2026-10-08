import cv2
import os
import glob
import math
import numpy as np


MODALITIES = [
    "xray",
    "ct",
    "mri"
]

BASE_DIR = "data/candidates"


for modality in MODALITIES:

    folder = os.path.join(
        BASE_DIR,
        modality
    )

    files = sorted(
        glob.glob(
            os.path.join(folder, "*.png")
        )
    )

    print(
        f"{modality.upper()}: {len(files)} images"
    )

    if not files:
        continue

    thumbnails = []

    for file in files:

        image = cv2.imread(
            file,
            cv2.IMREAD_GRAYSCALE
        )

        # Make thumbnail
        thumbnail = cv2.resize(
            image,
            (180, 180)
        )

        # Convert grayscale to BGR
        thumbnail = cv2.cvtColor(
            thumbnail,
            cv2.COLOR_GRAY2BGR
        )

        # Add filename/index
        name = os.path.basename(file)

        cv2.putText(
            thumbnail,
            name[-18:],
            (5, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

        thumbnails.append(thumbnail)

    columns = 5
    rows = math.ceil(
        len(thumbnails) / columns
    )

    canvas = np.zeros(
    (rows * 180, columns * 180, 3),
    dtype=np.uint8
)

    for i, thumbnail in enumerate(thumbnails):

        row = i // columns
        col = i % columns

        y = row * 180
        x = col * 180

        canvas[
            y:y + 180,
            x:x + 180
        ] = thumbnail

    output = f"{modality}_candidates.jpg"

    cv2.imwrite(
        output,
        canvas
    )

    print(
        f"Saved preview: {output}"
    )