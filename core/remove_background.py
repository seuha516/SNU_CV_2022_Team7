import sys
import os
import numpy as np
import cv2
from PIL import Image
from rembg import remove

BACKGROUND_DEFAULT = "data/default_background.jpg"
RESULT_DIR = "result/remove_background/"


def naive_interpolation(image : np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    background_resized = cv2.resize(background, (image.shape[1], image.shape[0]))
    background_resized = cv2.cvtColor(background_resized, cv2.COLOR_RGB2RGBA)

    # interpolate between image and background; weight is given from mask
    for x, y in np.ndindex(mask.shape):
        image_weight = mask[x, y] / 255 # value between 0 and 1
        result[x, y, :] = (1-image_weight) * background_resized[x, y, :] + image_weight * image[x, y, :]
    
    return result


def remove_background(image : np.ndarray, background : np.ndarray) -> np.ndarray:
    if background is None:
        return image

    image = np.array(Image.fromarray(image.astype(np.uint8)))

    # remove background
    bg_removed = remove(
        image,
        alpha_matting=True,
        alpha_matting_foreground_threshold=250,
        alpha_matting_background_threshold=60,
        alpha_matting_erode_size=10
    )

    # fill background with naive interpolation
    mask = remove(image, only_mask=True)
    return naive_interpolation(bg_removed, background, mask)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    image = cv2.imread(sys.argv[1])
    image = np.array(Image.fromarray(image.astype(np.uint8)))

    # remove background
    bg_removed = remove(
        image,
        alpha_matting=True,
        alpha_matting_foreground_threshold=250,
        alpha_matting_background_threshold=60,
        alpha_matting_erode_size=10
    )
    cv2.imwrite(os.path.join(RESULT_DIR, 'without_background.png'), bg_removed)

    background = cv2.imread(BACKGROUND_DEFAULT)
    mask = remove(image, only_mask=True)
    result = naive_interpolation(bg_removed, background, mask)
    cv2.imwrite(os.path.join(RESULT_DIR, 'new_background.png'), result)


if __name__ == '__main__':
    main()
