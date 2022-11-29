import numpy as np
from typing import Sequence


def fill(shape : Sequence[int], color : Sequence[int]) -> np.array:
    """Fill the image with the given color."""
    return np.full(shape, color, dtype=np.uint8)

def blend_naive(image : np.array, image2 : np.array, alpha : float = .9) -> np.array:
    """Blend two images using naive mode."""
    return alpha * image + (1 - alpha) * image2

def contrast(image : np.array, value : float) -> np.array:
    """Adjust the contrast of the image."""
    return (image - 127.5) * value + 127.5

def brightness(image : np.array, value : float) -> np.array:
    """Adjust the brightness of the image."""
    return image * value
