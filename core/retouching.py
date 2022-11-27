import numpy as np
import cv2 as cv
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from core.ehanet import *


def retouching(img=None, h=10, hColor=10, shape=None):
    dst = cv.fastNlMeansDenoisingColored(img, None, h, hColor, 7, 21)

    ###Retouching w/o mask###
    cv.imwrite('result/retouched_vanilla.jpg', dst)

    ###Detect the mask using cv###
    mask_poly = np.zeros((img.shape[0], img.shape[1], 1))

    l_eye = None
    r_eye = None
    mouth = None

    # Detect the left eye
    for i in range(36, 41):
        if l_eye is None:
            l_eye = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            l_eye = np.concatenate((l_eye, np.array(
                [[shape.part(i).x, shape.part(i).y]])), axis=0)
    mask_poly = cv.fillPoly(mask_poly, np.int32([l_eye]), 1)

    # Detect the right eye
    for i in range(42, 47):
        if r_eye is None:
            r_eye = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            r_eye = np.concatenate((r_eye, np.array(
                [[shape.part(i).x, shape.part(i).y]])), axis=0)
    mask_poly = cv.fillPoly(mask_poly, np.int32([r_eye]), 1)

    # Detect the mouth
    for i in range(48, 59):
        if mouth is None:
            mouth = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            mouth = np.concatenate((mouth, np.array(
                [[shape.part(i).x, shape.part(i).y]])), axis=0)

    mask_poly = cv.fillPoly(mask_poly, np.int32([mouth]), 1)

    cv.imwrite('result/cvmask.jpg', mask_poly*255)

    dst_poly = img*mask_poly + dst*(1-mask_poly)
    cv.imwrite('result/retouched_cvmask.jpg', dst_poly)

    ###Detect the mask using sementic segmentation###
    # convert CV to PIL image
    mat_img_origin = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    mat_img = mat_img_origin.resize((1024, 1024))

    # Preprocessing
    transform_image = transforms.Compose([transforms.Resize([256, 256]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])
                                          ])
    image_t = transform_image(mat_img)
    image_t = image_t[None]

    # Segmentation
    segmentator = EHANet18(num_classes=19, pretrained=True)
    segmentator.eval()

    mask_pred = PostProcessing(
        segmentator(PreProcessing(image_t))[0][-1])
    skin = mask_pred[0][0]

    # make mask
    mask = F.interpolate(torch.unsqueeze(torch.unsqueeze(
        skin, 0), 0), size=(img.shape[0], img.shape[1]), mode='bilinear')
    mask = mask.squeeze().unsqueeze(2)
    mask = mask.numpy()
    cv.imwrite('result/segmask.jpg', mask*255)

    dst = img*(1-mask) + dst*(mask)
    cv.imwrite('result/retouched_segmask.jpg', dst)

    return dst
