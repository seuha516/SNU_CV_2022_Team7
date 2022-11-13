import cv2
from tensorflow import keras

def classifying(img=None):
    face = cv2.resize(img, (128, 128))
    model = keras.models.load_model('./best-cnn-model.h5')
    result = list(model.predict(face.reshape(1, 128, 128, 1)[0:1])[0])

    emotion_list = ['기쁨', '당황', '분노', '슬픔', '중립']
    for emotion in emotion_list:
        print('{}: {}%'.format(emotion, result[emotion_list.index(emotion)] * 100))