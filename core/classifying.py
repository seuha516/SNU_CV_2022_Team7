import sys
import os
import cv2
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

RESULT_DIR = "result/classifying/"


def classifying(image):
    face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    face = face.reshape(-1, 128, 128, 1) / 255.0

    model = keras.models.load_model('best-cnn-model.h5')
    result = model.predict(face)[0]

    ret = "[ Facial Expression Classifying ]\n"
    emotion_list = ['Joy', 'Embarrassed', 'Anger', 'Sad', 'Neutral']
    for emotion in emotion_list:
        ret += '{}: {}%\n'.format(emotion, round(result[emotion_list.index(emotion)] * 100, 4))

    return ret


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    image = cv2.imread(sys.argv[1])
    result = classifying(image)

    with open(os.path.join(RESULT_DIR, 'facial_expression_classifying.txt'), 'w') as f:
        f.write(result)


if __name__ == '__main__':
    main()
