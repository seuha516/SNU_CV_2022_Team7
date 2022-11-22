import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def retouching(img=None, h=5, hColor=5, shape=None):
    dst = cv.fastNlMeansDenoisingColored(img, None, h, hColor, 7, 21)
    cv.imwrite('result/retouched_wo_mask.jpg', dst)

    mask = np.zeros((img.shape[0], img.shape[1], 1))
    print(mask.shape)
    l_eye = None
    r_eye = None
    mouth = None
    for i in range(36, 41):
        if l_eye is None:
            l_eye = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            l_eye = np.concatenate((l_eye, np.array(
                [[shape.part(i).x, shape.part(i).y]])), axis=0)
    mask = cv.fillPoly(mask, np.int32([l_eye]), 1)

    for i in range(42, 47):
        if r_eye is None:
            r_eye = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            r_eye = np.concatenate((r_eye, np.array(
                [[shape.part(i).x, shape.part(i).y]])), axis=0)
    mask = cv.fillPoly(mask, np.int32([r_eye]), 1)

    for i in range(48, 59):
        if mouth is None:
            mouth = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            mouth = np.concatenate((mouth, np.array(
                [[shape.part(i).x, shape.part(i).y]])), axis=0)
    mask = cv.fillPoly(mask, np.int32([mouth]), 1)

    cv.imwrite('result/mask.jpg', mask)

    dst = img*mask + dst*(1-mask)
    cv.imwrite('result/retouched_w_mask.jpg', dst)

    # dst /= 255.
    # b, g, r = cv.split(img)
    # img_show = cv.merge([r, g, b])
    # b, g, r = cv.split(dst)
    # dst_show = cv.merge([r, g, b])
    # plt.subplot(121), plt.imshow(img_show)
    # plt.subplot(122), plt.imshow(dst_show)
    # plt.show()

    return dst
