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

    # l_eye_upper = min(shape.part(37).y, shape.part(38).y)
    # l_eye_lower = max(shape.part(40).y, shape.part(41).y)
    # l_eye_left = shape.part(36).x
    # l_eye_right = shape.part(39).x
    # mask[l_eye_upper:l_eye_lower, l_eye_left:l_eye_right, :] = 1

    # r_eye_upper = min(shape.part(43).y, shape.part(44).y)
    # r_eye_lower = max(shape.part(46).y, shape.part(47).y)
    # r_eye_left = shape.part(42).x
    # r_eye_right = shape.part(45).x
    # mask[r_eye_upper:r_eye_lower, r_eye_left:r_eye_right, :] = 1.

    # mouth_upper = min(shape.part(50).y, shape.part(52).y)
    # mouth_lower = max(shape.part(56).y, shape.part(58).y)
    # mouth_left = shape.part(48).x
    # mouth_right = shape.part(54).x
    # mask[mouth_upper:mouth_lower, mouth_left:mouth_right, :] = 1.

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
