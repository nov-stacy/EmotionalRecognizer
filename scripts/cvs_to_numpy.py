from config.constants import SIZE_FACE, EMOTIONS, SAVE_DATASET_IMAGES_FILENAME, \
    SAVE_DATASET_LABELS_FILENAME, DATASET_CSV_FILENAME

import cv2
import pandas as pd
import numpy as np
from PIL import Image


cascade_path = '../venv/data/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(cascade_path)


def format_image(image_matrix):
    """
    Метод для обработки матрицы изображения для нейронной сети
    :param image_matrix: изображение в виде матрицы
    :return: обработанное изображение в виде матрицы
    """

    # преобразование в градации серого
    if len(image_matrix.shape) > 2 and image_matrix.shape[2] == 3:
        image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
    else:
        image_matrix = cv2.imdecode(image_matrix, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    gray_border = np.zeros((150, 150), np.uint8)
    gray_border[:, :] = 200
    gray_border[
        int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2)),
        int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2))
    ] = image_matrix
    image_matrix = gray_border

    # Обнаруживает объекты различных размеров на входном изображении.
    # Обнаруженные объекты возвращаются в виде списка прямоугольников.
    faces = cascade_classifier.detectMultiScale(image_matrix, scaleFactor=1.3, minNeighbors=5)

    # если списка нет, то значит лицо не было обнаружено на видео, и с этим изображением нейронной сети работать не надо
    if not len(faces) > 0:
        return None

    # нахождение максимального найденного объекта на изображении
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    # изменение размера изображения для обработки нейронной сетью
    image_matrix = image_matrix[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    image_matrix = cv2.resize(image_matrix, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC) / 255.

    return image_matrix


def emotion_to_vec(index_value):
    """
    Метод для преобразования эмоции в вектор
    :param index_value: значение эмоции (int)
    :return: вектор значений эмоций
    """
    vector = np.zeros(len(EMOTIONS))  # вектор из нулей
    vector[index_value] = 1.0  # истинная эмоция 1
    return vector


def data_to_image(main_data):
    """
    Метод для обработки данных для тренировки нейронной сети
    :param main_data: исходные данные (в виде строки)
    :return: обработанное изображение
    """
    # кодировка данных из строки в матрицу
    data_image = np.fromstring(str(main_data), dtype=np.uint8, sep=' ').reshape((SIZE_FACE, SIZE_FACE))
    # получение изображения из матрицы и конвертация с учетом RGB
    global a
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()  # создание из каждого значения x -> [x, x, x]
    data_image = format_image(data_image)  # обработка
    return data_image


if __name__ == '__main__':

    data = pd.read_csv('../data/' + DATASET_CSV_FILENAME)  # чтение датасета из памяти
    images, labels = [], []  # признаки и целевые значения
    for index, row in data.iterrows():
        emotion = emotion_to_vec(row['emotion'])  # обработка эмоции
        image = data_to_image(row['pixels'])  # обработка изображения
        if image is not None:
            labels.append(emotion)
            images.append(image)
        # вывод информации по процессу
        print("Progress: {}/{} {:.2f}%".format(index, data.shape[0], len(images) * 100.0 / data.shape[0]))

    # сохранение результатов работы
    np.save('../data/' + SAVE_DATASET_IMAGES_FILENAME, images)
    np.save('../data/' + SAVE_DATASET_LABELS_FILENAME, labels)