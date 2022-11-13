import os
import cv2
import numpy as np
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_SIZE = 128
train_path = '../preprocessed_data_for_cnn/train'
validation_path = '../preprocessed_data_for_cnn/validation'
emotion_list = ['joy', 'embarrassed', 'anger', 'sad', 'neutral']

train_images = []
train_answers = []
for emotion in emotion_list:
    path = os.path.join(train_path, emotion)
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        train_images.append(image)
        train_answers.append(emotion_list.index(emotion))
train_images = np.array(train_images)
train_answers = np.array(train_answers)
train_index = np.arange(train_answers.shape[0])
np.random.shuffle(train_index)
train_images = train_images[train_index]
train_answers = train_answers[train_index]
train_scaled = train_images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
print(train_scaled.shape)

validation_images = []
validation_answers = []
for emotion in emotion_list:
    path = os.path.join(validation_path, emotion)
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        validation_images.append(image)
        validation_answers.append(emotion_list.index(emotion))
validation_images = np.array(validation_images)
validation_answers = np.array(validation_answers)
validation_index = np.arange(validation_answers.shape[0])
np.random.shuffle(validation_index)
validation_images = validation_images[validation_index]
validation_answers = validation_answers[validation_index]
validation_scaled = validation_images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
print(validation_scaled.shape)

model = keras.Sequential()
model.add(keras.layers.Conv2D(
    32,
    kernel_size=3,
    activation='relu',
    padding='same',
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)
))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(
    64,
    kernel_size=3,
    activation='relu',
    padding='same',
))
model.add(keras.layers.Conv2D(
    64,
    kernel_size=3,
    activation='relu',
    padding='same',
))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(
    128,
    kernel_size=3,
    activation='relu',
    padding='same',
))
model.add(keras.layers.Conv2D(
    128,
    kernel_size=3,
    activation='relu',
    padding='same',
))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(5, activation='softmax'))
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)
checkpoint_cb = keras.callbacks.ModelCheckpoint('../best-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(
    train_scaled,
    train_answers,
    epochs=30,
    validation_data=(validation_scaled, validation_answers),
    callbacks=[checkpoint_cb, early_stopping_cb]
)