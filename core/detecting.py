import sys
import os
import cv2
import dlib

RESULT_DIR = "result/detecting/"


def detecting(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    detect = detector(image, 1)[0]
    face = image[detect.top() : detect.bottom(),  detect.left() : detect.right()]
    shape = predictor(image, detect)

    return face, shape


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    image = cv2.imread(sys.argv[1])
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    detect = detector(image, 1)[0]
    face = image[detect.top() : detect.bottom(),  detect.left() : detect.right()]
    shape = predictor(image, detect)
    cv2.imwrite(os.path.join(RESULT_DIR, 'face.png'), face)

    cv2.rectangle(image, (detect.left(), detect.top()), (detect.right(), detect.bottom()), (0, 255, 0), 2)
    for i in range(shape.num_parts):
        point = shape.part(i)
        cv2.circle(image, (point.x, point.y), 2, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(RESULT_DIR, 'result.png'), image)


if __name__ == '__main__':
    main()