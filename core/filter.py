import cv2
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline

from filter_base import *

RESULT_DIR = "./result/filters/"


def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))


def filter(image : np.ndarray, type : str) -> np.ndarray:
    if type == "summer":
        return summer_filter(image)
    elif type == "winter":
        return winter_filter(image)
    elif type == "walden":
        return walden_filter(image)
    elif type == "pencil":
        return pencil_filter(image)
    elif type == "nashville":
        return nashville_filter(image)
    else:
        return image


def summer_filter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    return sum


def winter_filter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    return win


def walden_filter(image : np.ndarray) -> np.ndarray:
    cs = fill(image.shape, [0, 68, 204])
    cs = blend_screen(image, cs)
    
    cr = brightness(cs, 1.1)
    cr = contrast(cr, -10)
    cr = sepia(cr, .3)
    cr = saturate(cr, 1.6)

    return cr

def nashville_filter(image : np.ndarray) -> np.ndarray:
    cs1 = fill(image.shape, [247, 176, 153])
    cm1 = blend_darken(image, cs1)

    cs2 = fill(image.shape, [0, 70, 150])
    cr = blend_lighten(cm1, cs2)

    cr = sepia(cr, .2)
    cr = contrast(cr, 1.2)
    cr = brightness(cr, 1.05)
    cr = saturate(cr, 1.2)

    return cr


def pencil_filter(img):
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_color    



def main():
    image_path = sys.argv[1]
    filter_type = sys.argv[2]
    image = cv2.imread(image_path)

    # apply filter
    result = filter(image, filter_type)
    cv2.imwrite(RESULT_DIR + f"{filter_type}.png", result)

if __name__ == "__main__":
    main()