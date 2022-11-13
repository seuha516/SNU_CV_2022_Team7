import cv2 as cv
from matplotlib import pyplot as plt


def retouching(img=None, h=5, hColor=5, shape=None):
    dst = cv.fastNlMeansDenoisingColored(img, None, h, hColor, 7, 21)

    b, g, r = cv.split(img)
    img_show = cv.merge([r, g, b])
    b, g, r = cv.split(dst)
    dst_show = cv.merge([r, g, b])

    plt.subplot(121), plt.imshow(img_show)
    plt.subplot(122), plt.imshow(dst_show)
    plt.show()

    return dst
