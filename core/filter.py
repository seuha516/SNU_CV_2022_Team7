import sys
import os
import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import UnivariateSpline

RESULT_DIR = "result/filter/"


def filter(image : np.ndarray, filter_type : str) -> np.ndarray:
    image = np.array(Image.fromarray(image.astype(np.uint8)))
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    if filter_type == "summer":
        return summer_filter(image)
    elif filter_type == "winter":
        return winter_filter(image)
    elif filter_type == "bright":
        return bright_filter(image)
    elif filter_type == "sepia":
        return sepia_filter(image)
    elif filter_type == "sunset":
        return sunset_filter(image)
    else:
        return image

def summer_filter(img):
    increaseLookupTable = _LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = _LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel, red_channel  = cv2.split(img)
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
    os.makedirs(RESULT_DIR, exist_ok=True)

    image = cv2.imread(sys.argv[1])
    for filter_type in [None, 'summer', 'winter', 'bright', 'sepia', 'sunset']:
        result = filter(image=image, filter_type=filter_type)
        cv2.imwrite(os.path.join(
            RESULT_DIR,
            f'{"original" if filter_type is None else filter_type}.png'
        ), result)


if __name__ == "__main__":
    from filter_base import *
    main()
else:
    from core.filter_base import *
