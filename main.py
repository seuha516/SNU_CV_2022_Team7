import os
import cv2
import argparse
from core.detecting import detecting
from core.classifying import classifying
from core.retouching import retouching
from core.remove_background import remove_background


def main(args):
    image = cv2.imread(args.image)
    background = cv2.imread(args.background)

    ### Face detecting ###
    face, shape = detecting(img=image)

    ### Facial expression classifying ###
    classifying(img=face)

    ### Retouching ###
    retouched_output = retouching(
        img=image, h=args.h, hColor=args.hColor, shape=shape)

    ### Remove background ###
    background_filled = remove_background(
        image=retouched_output, background=background)
    cv2.imwrite(os.path.join(args.result_dir,
                'bg_filled_output.png'), background_filled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='data/1.jpg',
                        help='image file')
    parser.add_argument('--background', type=str, default='data/sea.jpeg',
                        help='background file')
    parser.add_argument('--result_dir', type=str, default='result',
                        help='result directory')
    parser.add_argument('--h', type=int, default=10,
                        help='filter strength')
    parser.add_argument('--hColor', type=int, default=10,
                        help='filter strength for color components')
    args = parser.parse_args()
    main(args)
