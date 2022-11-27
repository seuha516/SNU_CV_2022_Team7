import dlib


def detecting(img=None):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    detect = detector(img, 1)[0]
    face = img[detect.top() : detect.bottom(),  detect.left() : detect.right()]
    shape = predictor(img, detect)

    return face, shape

