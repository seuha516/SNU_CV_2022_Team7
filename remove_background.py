import sys
import cv2
import numpy as np
from rembg import remove


def fill_background(image : np.ndarray, background: np.ndarray) -> np.ndarray:
    background_resized = cv2.resize(background, (image.shape[1], image.shape[0]))

    result = image.copy()

    # find where all 3 channels of image are 0
    # then fill with background_resized with the same coordinates
    for x, y in np.argwhere(np.all(image == 0, axis=-1)):
        result[x, y, :] = background_resized[x, y, :]
    
    return result


def main():
    image_path = sys.argv[1]
    background_path = sys.argv[2] if len(sys.argv) > 2 else None
    image = cv2.imread(image_path)

    result = remove(image)   
    cv2.imwrite("result/background_removed.png", result)

    if background_path is not None:
        image = cv2.imread("result/background_removed.png")
        background = cv2.imread(background_path)
        result = fill_background(image, background)

        # write array to image
        cv2.imwrite("result/background_filled.png", result)

if __name__ == '__main__':
    main()