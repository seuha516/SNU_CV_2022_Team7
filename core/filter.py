import cv2
import sys
import numpy as np
from scipy.interpolate import UnivariateSpline

from filter_base import *

RESULT_DIR = "./result/filters/"


def filter(image : np.ndarray, type : str) -> np.ndarray:
    if type == "summer":
        return summer_filter(image)
    elif type == "winter":
        return winter_filter(image)
    elif type == "bright":
        return bright_filter(image)
    elif type == "sepia":
        return sepia_filter(image)
    elif type == "sunset":
        return sunset_filter(image)
    else:
        return image


def summer_filter(img):
    increaseLookupTable = _LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = _LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    summer = cv2.merge((blue_channel, green_channel, red_channel ))
    return summer

def winter_filter(img):
    increaseLookupTable = _LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = _LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    winter = cv2.merge((blue_channel, green_channel, red_channel))
    return winter

def sunset_filter(img):
    increaseLookupTable = _LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    green_channel = cv2.LUT(green_channel, increaseLookupTable).astype(np.uint8)
    green_channel = cv2.LUT(green_channel, increaseLookupTable).astype(np.uint8)
    sunset = cv2.merge((blue_channel, green_channel, red_channel))
    sunset = brightness(sunset, .9)
    return sunset

def bright_filter(img):
    increaseLookupTable = _LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    green_channel = cv2.LUT(green_channel, increaseLookupTable).astype(np.uint8)
    bright = cv2.merge((blue_channel, green_channel, red_channel))
    return bright

def sepia_filter(img):
    bg = fill(img.shape, (112, 66, 20))
    result = blend_naive(img, bg, .7)
    result = brightness(result, 1.2)
    return result

def _LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))


def main():
    image_path = sys.argv[1]
    filter_type = sys.argv[2]
    image = cv2.imread(image_path)

    # apply filter
    result = filter(image, filter_type)
    cv2.imwrite(RESULT_DIR + f"{filter_type}.png", result)

if __name__ == "__main__":
    main()