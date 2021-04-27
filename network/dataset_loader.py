from config.constants import SIZE_FACE, EMOTIONS, SAVE_DATASET_IMAGES_FILENAME, SAVE_DATASET_LABELS_FILENAME

import numpy as np
from sklearn.model_selection import train_test_split


# класс для загрузки датасета и разбиения ее на тестовую и тренировочную выборку
class DatasetLoader:

    __images__, __labels__ = None, None  # признаки и целевые переменные

    @classmethod
    def load(cls, directory='../data/'):
        """
        Метод для загрузки датасета из памяти
        """
        # загрузка признаков
        cls.__images__ = np.load(directory + SAVE_DATASET_IMAGES_FILENAME).reshape([-1, SIZE_FACE, SIZE_FACE, 1])
        # загрузка целевой переменной
        cls.__labels__ = np.load(directory + SAVE_DATASET_LABELS_FILENAME).reshape([-1, len(EMOTIONS)])

    @classmethod
    def train_test_split(cls):
        """
        Метод для разбиения выборки на тестовую и тренировочную выборки в отношении 80:20
        :return: тестовую и тренировочную выборки (x_train, y_train, x_test, y_test)
        """
        return train_test_split(cls.__images__, cls.__labels__, test_size=0.2, random_state=42)

    @classmethod
    def images(cls):
        """
        Геттер для признаков из выборки
        """
        return cls.__images__

    @classmethod
    def labels(cls):
        """
        Геттер для целевой переменной из выборки
        """
        return cls.__labels__

