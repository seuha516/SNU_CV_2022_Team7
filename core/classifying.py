import os
import cv2
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def classifying(img=None):
    face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    face = face.reshape(-1, 128, 128, 1) / 255.0

    model = keras.models.load_model('./best-cnn-model.h5')
    result = model.predict(face)[0]

    emotion_list = ['기쁨', '당황', '분노', '슬픔', '중립']
    for emotion in emotion_list:
        print('{}: {}%'.format(emotion, round(result[emotion_list.index(emotion)] * 100, 4)))