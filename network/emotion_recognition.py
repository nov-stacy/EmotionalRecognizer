from config.constants import SIZE_FACE, EMOTIONS, SAVE_MODEL_FILENAME

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tensorflow import logging
import os

from network.dataset_loader import DatasetLoader

# убирает все WARNING sот TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.set_verbosity(logging.ERROR)


# класс для описания нейронной сети, которая будет определять эмоции по изображению
class EmotionRecognition:

    def __init__(self):
        self.__model__ = None  # инициализация модели
        self.__build_network__()  # создание модели

    def __build_network__(self):
        """
        Метод для генерации каркаса модели
        """
        print('[+] Start building DNN')  # отправка в консоль сообщения об успешном старте создания каркаса
        _network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1])
        _network = conv_2d(_network, 64, 5, activation='relu')
        _network = max_pool_2d(_network, 3, strides=2)
        _network = conv_2d(_network, 64, 5, activation='relu')
        _network = max_pool_2d(_network, 3, strides=2)
        _network = conv_2d(_network, 128, 4, activation='relu')
        _network = dropout(_network, 0.3)
        _network = fully_connected(_network, 3072, activation='relu')
        _network = fully_connected(_network, len(EMOTIONS), activation='softmax')
        _network = regression(_network, optimizer='momentum', metric='accuracy', loss='categorical_crossentropy')
        self.__model__ = tflearn.DNN(_network, checkpoint_path='emotion_recognition', max_checkpoints=1,
                                     tensorboard_verbose=2)
        print('[+] Stop building DNN')  # отправка в консоль сообщения об успешном создании каркаса

    def __save_model__(self):
        """
        Метод для сохранения модели
        """
        self.__model__.save(SAVE_MODEL_FILENAME)
        print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)

    def load_model(self, directory='network/'):
        """
        Метод для загрузки модели (в частности весов) из памяти
        :param directory: директория, в которой находится файл с моделью
        """
        # проверка на наличие файлов в директории
        if os.path.isfile(directory + SAVE_MODEL_FILENAME + '.meta'):
            self.__model__.load(directory + SAVE_MODEL_FILENAME)  # загрузка модели
            print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)  # отправка в консоль сообщения об успешной загрузке
        # если не найдена модель, то будет вызвано исключение, так как без модели нейронка не работает
        else:
            raise ValueError

    def predict(self, image):
        """
        Метод для предсказания эмоций по изображению
        :param image: Изображение
        """
        return self.__model__.predict(image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])) if image is not None else None

    def train(self):
        """
        Метод для тренировки модели
        """
        # получение тренировочной и тестовой выборки
        x_train, y_train, x_test, y_test = DatasetLoader.train_test_split()

        print('[+] Start training network')  # отправка в консоль сообщения об успешно старте тренировки модели
        # тренировка модели
        self.__model__.fit(
            x_train, y_train,
            validation_set=(x_test, y_test),
            n_epoch=100,
            batch_size=50,
            shuffle=True,
            show_metric=True,
            snapshot_step=200,
            snapshot_epoch=True,
            run_id='emotion_recognition'
        )
        print('[+] Stop training network')  # отправка в консоль сообщения об успешной тренировки модели
        self.__save_model__()  # сохранение модели


if __name__ == "__main__":
    network = EmotionRecognition()
    network.train()
