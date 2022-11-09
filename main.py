import sys
import dlib
import cv2
import numpy as np
from core.retouching import retouching
import argparse


def main(args):
    # image_path = sys.argv[1]
    # background_path = sys.argv[2] if len(sys.argv) > 2 else None
    image = cv2.imread(args.image_dir)

    # TODO... FACE DETECTION
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    face = detector(image, 1)[0]
    # cv2.rectangle(image, (face.left(), face.top()),
    #               (face.right(), face.bottom()), (0, 255, 0), 2)

    shape = predictor(image, face)

    ###Retouching###
    image = retouching(img=image, h=args.h, shape=shape)

    for i in range(shape.num_parts):
        point = shape.part(i)
        cv2.circle(image, (point.x, point.y), 2, (0, 0, 255), 2)

    cv2.imwrite(args.result_dir, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/1.jpg',
                        help='image directory')
    parser.add_argument('--background_dir', type=str, default='data/background.jpg',
                        help='background directory')
    parser.add_argument('--result_dir', type=str, default='result/output.png',
                        help='result directory')
    parser.add_argument('--h', type=int, default=5,
                        help='filter strength')
    parser.add_argument('--hColor', type=int, default=5,
                        help='filter strength for color components')

    args = parser.parse_args()
    main(args)
