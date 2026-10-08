import cv2
import numpy as np


TARGET_SIZE = (255, 255)


def load_image(image_source):
    """
    Load an image from a file path or Streamlit uploaded file.

    Returns:
        BGR image as a NumPy array.
    """

    if isinstance(image_source, str):
        image = cv2.imread(image_source)

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_source}"
            )

        return image

    # Streamlit UploadedFile
    file_bytes = np.asarray(
        bytearray(image_source.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode uploaded image.")

    return image


def resize_image(image, size=TARGET_SIZE):
    """
    Resize image to the resolution used in the research paper.
    """

    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )


def convert_to_grayscale(image):
    """
    Convert BGR/RGB image to 8-bit grayscale.
    """

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def preprocess_image(image):
    """
    Complete preprocessing pipeline:

    1. Resize to 255 x 255
    2. Convert to grayscale
    3. Ensure uint8 format
    """

    resized = resize_image(image)

    gray = convert_to_grayscale(resized)

    gray = np.clip(
        gray,
        0,
        255
    ).astype(np.uint8)

    return gray