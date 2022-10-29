import sys
import dlib
import cv2
import numpy as np

def main():
    image_path = sys.argv[1]
    background_path = sys.argv[2] if len(sys.argv) > 2 else None
    image = cv2.imread(image_path)

    # TODO... FACE DETECTION
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    face = detector(image, 1)[0]
    cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

    shape = predictor(image, face)
    for i in range(shape.num_parts):
        point = shape.part(i)
        cv2.circle(image, (point.x, point.y), 2, (0, 0, 255), 2)

    cv2.imwrite('result/output.png', image)


if __name__ == '__main__':
    main()