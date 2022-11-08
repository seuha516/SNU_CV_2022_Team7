import sys
import cv2
import numpy as np
from rembg import remove

def fill_background(image : np.ndarray, background: np.ndarray, mask: np.ndarray, mask_threshold : int) -> np.ndarray:
    background_resized = cv2.resize(background, (image.shape[1], image.shape[0]))

    result = image.copy()

    # interpolate between image and background; weight is given from mask
    for x, y in np.argwhere(mask <= mask_threshold):
        image_weight = mask[x, y] / 255 # value between 0 and 1
        result[x, y, :] = (1-image_weight) * background_resized[x, y, :] + image_weight * image[x, y, :]
    
    return result


def main():
    image_path = sys.argv[1]
    background_path = sys.argv[2] if len(sys.argv) > 2 else None
    mask_threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 255 # mask.max() = 255
    image = cv2.imread(image_path)

    # remove background
    result = remove(image)   
    cv2.imwrite("result/background_removed.png", result)

    # fill background
    if background_path is not None:
        image = cv2.imread("result/background_removed.png")
        background = cv2.imread(background_path)
        mask = remove(image, only_mask=True)

        result = fill_background(image, background, mask, mask_threshold)

        cv2.imwrite(f"result/background_filled_mask_{mask_threshold}.png", result)

if __name__ == '__main__':
    main()