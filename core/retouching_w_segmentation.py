import numpy as np
import cv2 as cv
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from core.ehanet import *


def retouching(img=None, h=5, hColor=5, shape=None):
    dst = cv.fastNlMeansDenoisingColored(img, None, h, hColor, 7, 21)
    cv.imwrite('result/retouched_wo_mask.jpg', dst)

    mat_img_origin = Image.open('data/1.jpg')
    mat_img = mat_img_origin.resize((1024, 1024))

    transform_image = transforms.Compose([transforms.Resize([256, 256]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])
                                          ])
    image_t = transform_image(mat_img)
    image_t = image_t[None]

    segmentator = EHANet18(num_classes=19, pretrained=True)
    segmentator.eval()

    mask_pred = PostProcessing(
        segmentator(PreProcessing(image_t))[0][-1])
    skin = mask_pred[0][0]
    mask = F.interpolate(torch.unsqueeze(torch.unsqueeze(
        skin, 0), 0), size=(img.shape[0], img.shape[1]), mode='bilinear')
    mask = mask.squeeze().unsqueeze(2)
    mask = mask.numpy()
    cv.imwrite('result/mask_segmentation.jpg', mask)
    dst = img*(1-mask) + dst*(mask)
    cv.imwrite('result/retouched_w_mask_segmentation.jpg', dst)

    return dst
