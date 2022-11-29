import os
import cv2
import argparse
from core.detecting import detecting
from core.classifying import classifying
from core.retouching import retouching
from core.remove_background import remove_background
from core.filter import filter


def main(args):
    os.makedirs(args.result_dir, exist_ok=True)

    image = cv2.imread(args.image)
    background = cv2.imread(args.background)

    ### Face detecting ###
    print('Step 1 : Face detecting...')
    face, shape = detecting(image=image)
    print('Step 1 : Complete')

    ### Facial expression classifying ###
    print('Step 2 : Facial expression classifying...')
    result = classifying(image=face)
    with open(os.path.join(args.result_dir, 'facial_expression_classifying.txt'), 'w') as f:
        f.write(result)
    print('Step 2 : Complete')

    ### Retouching ###
    print('Step 3 : Retouching...')
    image = retouching(image=image, h=args.h, hColor=args.hColor, shape=shape)
    print('Step 3 : Complete')

    ### Remove background ###
    print('Step 4 : Remove background...')
    image = remove_background(image=image, background=background)
    print('Step 4 : Complete')

    ### Filter ###
    print('Step 5 : Filter...')
    for filter_type in [None, 'summer', 'winter', 'bright', 'sepia', 'sunset']:
        result = filter(image=image, filter_type=filter_type)
        cv2.imwrite(os.path.join(
            args.result_dir,
            f'{"original" if filter_type is None else filter_type}.png'
        ), result)
    print('Step 5 : Complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='data/default_selfie.jpg',
                        help='image file')
    parser.add_argument('--background', type=str, default=None,
                        help='background file')
    parser.add_argument('--result_dir', type=str, default='result',
                        help='result directory')
    parser.add_argument('--h', type=int, default=10,
                        help='reouching filter strength')
    parser.add_argument('--hColor', type=int, default=10,
                        help='reouching filter strength for color components')
    args = parser.parse_args()
    main(args)
