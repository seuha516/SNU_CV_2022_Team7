from PIL import Image
import numpy as np
import cv2
from typing import Sequence

from PIL import Image


def fill(shape : Sequence[int], color : Sequence[int]) -> np.array:
    """Fill the image with the given color."""
    return np.full(shape, color, dtype=np.uint8)


def blend_screen(image : np.array, image2 : np.array) -> np.array:
    """Blend two images using screen mode."""
    return 255 - (255 - image) * (255 - image2) / 255

def blend_darken(image : np.array, image2 : np.array) -> np.array:
    """Blend two images using darken mode."""
    return np.minimum(image, image2)

def blend_lighten(image : np.array, image2 : np.array) -> np.array:
    """Blend two images using lighten mode."""
    return np.maximum(image, image2)

def contrast(image : np.array, value : float) -> np.array:
    """Adjust the contrast of the image."""
    return (image - 127.5) * value + 127.5

def brightness(image : np.array, value : float) -> np.array:
    """Adjust the brightness of the image."""
    return image * value

def saturate(image : np.array, value : float) -> np.array:
    """Adjust the saturation of the image."""
    matrix = [
        0.213 + 0.787 * value,
        0.715 - 0.715 * value,
        0.072 - 0.072 * value,
        0.213 - 0.213 * value,
        0.715 + 0.285 * value,
        0.072 - 0.072 * value,
        0.213 - 0.213 * value,
        0.715 - 0.715 * value,
        0.072 + 0.928 * value,
    ]
    matrix = np.matrix(matrix).reshape(3, 3)
    return cv2.transform(image, matrix)

def sepia(image : np.array, value : float) -> np.array:
    """Apply a sepia filter to the image."""
    matrix = [
        0.393 + 0.607 * value,
        0.769 - 0.769 * value,
        0.189 - 0.189 * value,
        0.349 - 0.349 * value,
        0.686 + 0.314 * value,
        0.168 - 0.168 * value,
        0.272 - 0.272 * value,
        0.534 - 0.534 * value,
        0.131 + 0.869 * value,
    ]
    matrix = np.matrix(matrix).reshape(3, 3)
    return cv2.transform(image, matrix)