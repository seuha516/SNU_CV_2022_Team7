import os, shutil, json, random
import cv2
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def sample_extraction(path, num):
    sample_index_list = random.sample(list(range(len(os.listdir(path)))), num)
    cnt = 0
    for image_name in os.listdir(path):
        if cnt not in sample_index_list:
            os.remove(os.path.join(path, image_name))
        cnt += 1


def main():
    # for train & validation. (12000 + 3000) * 5
    train_path = './data_for_cnn/Training'
    train_label_folder_format = '[라벨]EMOIMG_{}_TRAIN'
    train_label_file_format = 'img_emotion_training_data({}).json'
    train_image_folder_format = '[원천]EMOIMG_{}_TRAIN_01'

    # for test. 2000 * 5
    validation_path = './data_for_cnn/Validation'
    validation_label_folder_format = '[라벨]EMOIMG_{}_VALID'
    validation_label_file_format = 'img_emotion_validation_data({}).json'
    validation_image_folder_format = '[원천]EMOIMG_{}_VALID'

    # preprocessed data
    preprocessed_train_path = './preprocessed_data_for_cnn/train'
    preprocessed_validation_path = './preprocessed_data_for_cnn/validation'
    preprocessed_test_path = './preprocessed_data_for_cnn/test'

    # 5 facial expressions
    emotion_list = ['기쁨', '당황', '분노', '슬픔', '중립']
    emotion_eng_list = ['joy', 'embarrassed', 'anger', 'sad', 'neutral']
    emotion_eng = {
        '기쁨': 'joy',
        '당황': 'embarrassed',
        '분노': 'anger',
        '슬픔': 'sad',
        '중립': 'neutral'
    }

    # Sample Extraction
    for emotion in emotion_list:
        sample_extraction(
            os.path.join(
                train_path,
                train_image_folder_format.format(emotion)
            ),
            15000
        )
        sample_extraction(
            os.path.join(
                validation_path,
                validation_image_folder_format.format(emotion)
            ),
            2000
        )

    # Data Arrange
    for emotion in emotion_list:
        sample_index_list = random.sample(list(range(15000)), 3000)
        tp = os.path.join(
            train_path,
            train_image_folder_format.format(emotion)
        )

        cnt = 0
        for image_name in os.listdir(tp):
            if cnt in sample_index_list:
                shutil.move(
                    os.path.join(tp, image_name),
                    os.path.join(preprocessed_validation_path, emotion_eng[emotion], image_name)
                )
            else:
                shutil.move(
                    os.path.join(tp, image_name),
                    os.path.join(preprocessed_train_path, emotion_eng[emotion], image_name)
                )
            cnt += 1

        shutil.copy(
            os.path.join(
                train_path,
                train_label_folder_format.format(emotion),
                train_label_file_format.format(emotion)
            ),
            os.path.join(
                preprocessed_train_path,
                '{}.json'.format(emotion_eng[emotion])
            )
        )
        shutil.move(
            os.path.join(
                train_path,
                train_label_folder_format.format(emotion),
                train_label_file_format.format(emotion)
            ),
            os.path.join(
                preprocessed_validation_path,
                '{}.json'.format(emotion_eng[emotion])
            )
        )

        vp = os.path.join(
            validation_path,
            validation_image_folder_format.format(emotion)
        )
        for image_name in os.listdir(vp):
            shutil.move(
                os.path.join(vp, image_name),
                os.path.join(preprocessed_test_path, emotion_eng[emotion], image_name)
            )

        shutil.move(
            os.path.join(
                validation_path,
                validation_label_folder_format.format(emotion),
                validation_label_file_format.format(emotion)
            ),
            os.path.join(
                preprocessed_test_path,
                '{}.json'.format(emotion_eng[emotion])
            )
        )

    # Get Face Image
    face_size_file = open('./preprocessed_data_for_cnn/face_size.txt', 'w')
    for preprocessed_path in [
        preprocessed_train_path,
        preprocessed_validation_path,
        preprocessed_test_path
    ]:
        num = 0
        for emotion in emotion_eng_list:
            with open(
                os.path.join(
                    preprocessed_path,
                    '{}.json'.format(emotion)
                )
            ) as json_file:
                json_data = json.load(json_file)

            path = os.path.join(preprocessed_path, emotion)
            for image_name in os.listdir(path):
                image_path = os.path.join(path, str(num) + '.jpg')
                os.rename(os.path.join(path, image_name), image_path)
                image = cv2.imread(image_path)

                json_item = next((item for item in json_data if item['filename'] == image_name), None)
                face_box = json_item['annot_C']['boxes']

                minX = max(round(face_box['minX']), 0)
                minY = max(round(face_box['minY']), 0)
                maxX = min(round(face_box['maxX']), image.shape[1] - 1)
                maxY = min(round(face_box['maxY']), image.shape[0] - 1)

                image = image[minY:maxY, minX:maxX]
                cv2.imwrite(image_path, image)

                face_size_file.write('{} {}\n'.format(image.shape[1], image.shape[0]))
                num += 1

    # Image Resize
    for preprocessed_path in [
        preprocessed_train_path,
        preprocessed_validation_path,
        preprocessed_test_path
    ]:
        for emotion in emotion_eng_list:
            image_folder_path = os.path.join(preprocessed_path, emotion)
            for image_name in os.listdir(image_folder_path):
                image_path = os.path.join(image_folder_path, image_name)

                image = cv2.imread(image_path)
                if 616 <= image.shape[1] <= 1016 and 837 <= image.shape[0] <= 1357:
                    image = np.array(tf.image.resize_with_crop_or_pad(np.array(image), 1024, 1024))
                    cv2.imwrite(image_path, image)
                else:
                    os.remove(image_path)

    # Image Size Reduction
    for preprocessed_path in [
        preprocessed_train_path,
        preprocessed_validation_path,
        preprocessed_test_path
    ]:
        for emotion in emotion_eng_list:
            image_folder_path = os.path.join(preprocessed_path, emotion)
            for image_name in os.listdir(image_folder_path):
                image_path = os.path.join(image_folder_path, image_name)

                image = cv2.imread(image_path)
                image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(image_path, image)




if __name__ == '__main__':
    main()