import os
import cv2
import numpy as np
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_SIZE = 128
model_path = '../best-cnn-model.h5'
test_path = '../preprocessed_data_for_cnn/test'
emotion_list = ['joy', 'embarrassed', 'anger', 'sad', 'neutral']

test_images = []
test_answers = []
for emotion in emotion_list:
    path = os.path.join(test_path, emotion)
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        test_images.append(image)
        test_answers.append(emotion_list.index(emotion))
test_images = np.array(test_images)
test_answers = np.array(test_answers)
test_scaled = test_images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
print(test_scaled.shape)

model = keras.models.load_model(model_path)
model.evaluate(test_scaled, test_answers)