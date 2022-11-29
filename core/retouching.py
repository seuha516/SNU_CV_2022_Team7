import sys
import os
import numpy as np
import cv2 as cv
import dlib
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

RESULT_DIR = "result/retouching/"


def retouching(image, h, hColor, shape):
    dst = cv.fastNlMeansDenoisingColored(image, None, h, hColor, 7, 21)
    l_eye = None
    r_eye = None
    mouth = None

    # Detect the left eye
    for i in range(36, 41):
        if l_eye is None:
            l_eye = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            l_eye = np.concatenate((l_eye, np.array([[shape.part(i).x, shape.part(i).y]])), axis=0)

    # Detect the right eye
    for i in range(42, 47):
        if r_eye is None:
            r_eye = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            r_eye = np.concatenate((r_eye, np.array([[shape.part(i).x, shape.part(i).y]])), axis=0)

    # Detect the mouth
    for i in range(48, 59):
        if mouth is None:
            mouth = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            mouth = np.concatenate((mouth, np.array([[shape.part(i).x, shape.part(i).y]])), axis=0)

    ###Detect the mask using sementic segmentation###
    # convert CV to PIL image
    mat_img_origin = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    mat_img = mat_img_origin.resize((1024, 1024))

    # Preprocessing
    transform_image = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_t = transform_image(mat_img)
    image_t = image_t[None]

    # Segmentation
    segmentator = EHANet18(num_classes=19, pretrained=True)
    segmentator.eval()

    mask_pred = PostProcessing(segmentator(PreProcessing(image_t))[0][-1])
    skin = mask_pred[0][0]

    # make mask
    mask = F.interpolate(
        torch.unsqueeze(torch.unsqueeze(skin, 0), 0),
        size=(image.shape[0], image.shape[1]),
        mode='bilinear'
    )
    mask = mask.squeeze().unsqueeze(2)
    mask = mask.numpy()

    return image * (1 - mask) + dst * mask


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    image = cv.imread(sys.argv[1])
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    shape = predictor(image, detector(image, 1)[0])

    dst = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    cv.imwrite(os.path.join(RESULT_DIR, 'retouched_vanilla.jpg'), dst)

    mask_poly = np.zeros((image.shape[0], image.shape[1], 1))
    l_eye = None
    r_eye = None
    mouth = None

    for i in range(36, 41):
        if l_eye is None:
            l_eye = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            l_eye = np.concatenate((l_eye, np.array([[shape.part(i).x, shape.part(i).y]])), axis=0)
    mask_poly = cv.fillPoly(mask_poly, np.int32([l_eye]), 1)
    for i in range(42, 47):
        if r_eye is None:
            r_eye = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            r_eye = np.concatenate((r_eye, np.array([[shape.part(i).x, shape.part(i).y]])), axis=0)
    mask_poly = cv.fillPoly(mask_poly, np.int32([r_eye]), 1)
    for i in range(48, 59):
        if mouth is None:
            mouth = np.array([[shape.part(i).x, shape.part(i).y]])
        else:
            mouth = np.concatenate((mouth, np.array([[shape.part(i).x, shape.part(i).y]])), axis=0)
    mask_poly = cv.fillPoly(mask_poly, np.int32([mouth]), 1)
    cv.imwrite(os.path.join(RESULT_DIR, 'cvmask.jpg'), mask_poly * 255)

    dst_poly = image * mask_poly + dst * (1 - mask_poly)
    cv.imwrite(os.path.join(RESULT_DIR, 'retouched_cvmask.jpg'), dst_poly)

    mat_img_origin = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    mat_img = mat_img_origin.resize((1024, 1024))

    transform_image = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_t = transform_image(mat_img)
    image_t = image_t[None]

    segmentator = EHANet18(num_classes=19, pretrained=True)
    segmentator.eval()

    mask_pred = PostProcessing(segmentator(PreProcessing(image_t))[0][-1])
    skin = mask_pred[0][0]

    mask = F.interpolate(
        torch.unsqueeze(torch.unsqueeze(skin, 0), 0),
        size=(image.shape[0], image.shape[1]),
        mode='bilinear'
    )
    mask = mask.squeeze().unsqueeze(2)
    mask = mask.numpy()
    cv.imwrite(os.path.join(RESULT_DIR, 'segmask.jpg'), mask * 255)

    dst = image * (1 - mask) + dst * mask
    cv.imwrite(os.path.join(RESULT_DIR, 'retouched_segmask.jpg'), dst)


if __name__ == '__main__':
    from ehanet import *
    main()
else:
    from core.ehanet import *
