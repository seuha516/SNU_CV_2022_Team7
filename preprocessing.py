import os, shutil, json
import random
from PIL import Image

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
                    os.path.join(preprocessed_validation_path, emotion, image_name)
                )
            else:
                shutil.move(
                    os.path.join(tp, image_name),
                    os.path.join(preprocessed_train_path, emotion, image_name)
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
                '{}.json'.format(emotion)
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
                '{}.json'.format(emotion)
            )
        )

        vp = os.path.join(
            validation_path,
            validation_image_folder_format.format(emotion)
        )
        for image_name in os.listdir(vp):
            shutil.move(
                os.path.join(vp, image_name),
                os.path.join(preprocessed_test_path, emotion, image_name)
            )

        shutil.move(
            os.path.join(
                validation_path,
                validation_label_folder_format.format(emotion),
                validation_label_file_format.format(emotion)
            ),
            os.path.join(
                preprocessed_test_path,
                '{}.json'.format(emotion)
            )
        )

    # Face Detection
    face_size_file = open('./preprocessed_data_for_cnn/face_size.txt', 'w')
    for preprocessed_path in [
        preprocessed_train_path,
        preprocessed_validation_path,
        preprocessed_test_path
    ]:
        for emotion in emotion_list:
            with open(
                os.path.join(
                    preprocessed_path,
                    '{}.json'.format(emotion)
                )
            ) as json_file:
                json_data = json.load(json_file)

            path = os.path.join(preprocessed_path, emotion)
            for image_name in os.listdir(path):
                image_path = os.path.join(path, image_name)
                image = Image.open(image_path)

                json_item = next((item for item in json_data if item['filename'] == image_name), None)
                face_box = json_item['annot_C']['boxes']
                image = image.crop((face_box['minX'], face_box['minY'], face_box['maxX'], face_box['maxY']))
                image.save(image_path)

                face_size_file.write('{} {}\n'.format(image.size[0], image.size[1]))


if __name__ == '__main__':
    main()