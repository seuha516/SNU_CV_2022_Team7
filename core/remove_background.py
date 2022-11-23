import sys
import cv2
import numpy as np
from rembg import remove


RESULT_DIR = "../result/background/"

def naive_interpolation(image : np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    background_resized = cv2.resize(background, (image.shape[1], image.shape[0]))

    result = image.copy()

    # interpolate between image and background; weight is given from mask
    for x, y in np.ndindex(mask.shape):
        image_weight = mask[x, y] / 255 # value between 0 and 1
        result[x, y, :] = (1-image_weight) * background_resized[x, y, :] + image_weight * image[x, y, :]
    
    return result


def pyramid_blending(image : np.ndarray, background: np.ndarray, mask: np.ndarray, pyr_levels : int) -> np.ndarray:
    # resize background to image size
    background_resized = cv2.resize(background, (image.shape[1], image.shape[0]))


    # Gaussian pyramid for image
    G = image.copy()
    G_im = [G] # length 7: original image + 6 levels
    for _ in range(pyr_levels):
        G = cv2.pyrDown(G)
        G_im.append(G)

    # Gaussian pyramid for background
    G = background_resized.copy()
    G_bg = [G]
    for _ in range(pyr_levels):
        G = cv2.pyrDown(G)
        G_bg.append(G)

    # laplacian pyramid for image
    L_im = [] # length 7; last level is just the last element of G_im
    for i in range(pyr_levels):
        l_im = cv2.subtract(G_im[i], cv2.pyrUp(G_im[i+1], dstsize=(G_im[i].shape[1], G_im[i].shape[0])))
        L_im.append(l_im)
    L_im.append(G_im[-1])

    # laplacian pyramid for background
    L_bg = []
    for i in range(pyr_levels):
        l_bg = cv2.subtract(G_bg[i], cv2.pyrUp(G_bg[i+1], dstsize=(G_bg[i].shape[1], G_bg[i].shape[0])))
        L_bg.append(l_bg)
    L_bg.append(G_bg[-1])

    # combine two laplacian pyramids using mask from rembg as weight
    L_blend = []
    for l_im, l_bg in zip(L_im, L_bg):
        mask_resized = cv2.resize(mask, (l_im.shape[1], l_im.shape[0]), interpolation=cv2.INTER_AREA)
        l_blend = np.zeros_like(l_im)

        for x, y in np.ndindex(mask_resized.shape):
            image_weight = mask_resized[x, y] / 255 # value between 0 and 1
            l_blend[x, y, :] = (1-image_weight) * l_bg[x, y, :] + image_weight * l_im[x, y, :]
        
        L_blend.append(l_blend)

    # reconstruct image (start from smallest layer)
    blended = L_blend[-1]
    for i in range(len(L_blend)-2, -1, -1):
        blended = cv2.pyrUp(blended, dstsize=(L_blend[i].shape[1], L_blend[i].shape[0]))
        blended = cv2.add(blended, L_blend[i])

    return blended


## to be independently executed
def main():
    image_path = sys.argv[1]
    background_path = sys.argv[2] if len(sys.argv) > 2 else None
    pyr_levels = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    image = cv2.imread(image_path)

    # remove background
    result = remove(image, alpha_matting=True, alpha_matting_foreground_threshold=250, alpha_matting_background_threshold=60 ,alpha_matting_erode_size=10)   
    cv2.imwrite(RESULT_DIR + "background_removed.png", result)

    # fill background
    if background_path is not None:
        image = cv2.imread(RESULT_DIR + "background_removed.png")
        background = cv2.imread(background_path)
        mask = remove(image, only_mask=True)

        # result = naive_interpolation(image, background, mask)
        result = pyramid_blending(image, background, mask, pyr_levels)

        cv2.imwrite(RESULT_DIR + f"background_filled_pyramid_{pyr_levels}.png", result)


## to be imported in main.py
def remove_background(image : np.ndarray, background : np.ndarray) -> np.ndarray:
    # remove background
    bg_removed = remove(image, alpha_matting=True, alpha_matting_foreground_threshold=250, alpha_matting_background_threshold=60 ,alpha_matting_erode_size=10)
    # fill background with naive interpolation
    mask = remove(image, only_mask=True)
    bg_filled = naive_interpolation(bg_removed, background, mask)
    print("Successfully removed background and filled it with the given background image.")
    return bg_filled


if __name__ == '__main__':
    main()