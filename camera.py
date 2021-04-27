import cv2
import numpy as np
from config.constants import *
from network.emotion_recognition import EmotionRecognition

# классификатор OpenCV для распознавания лица на изображении
cascade_path = 'data/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(cascade_path)


def format_image(image):
    """
    Метод для обработки изображения с камеры для последующего определения на нем эмоции с помощью нейронной сети
    :param image: чистое изображение с web-камеры
    :return: None, если не было найдено лицо, иначе обработанное изображение для нейронной сети
    """

    # преобразование в градации серого
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    # Обнаруживает объекты различных размеров на входном изображении.
    # Обнаруженные объекты возвращаются в виде списка прямоугольников.
    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

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
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC) / 255.

    return image


# класс, описывающий объект камеры для получения изображения из видеоряда и дальнейшей ее обработки
class Camera:

    def __init__(self, file_name=None):
        """
        При инициализации объекта камеры захватывается изображение с web-камеры для дальнейшей работы
        """
        self.__video_capture__ = cv2.VideoCapture(0 if file_name is None else file_name)  # захват видео
        self.__need_flip__ = file_name is None
        self.__network__ = EmotionRecognition()  # загрузка нейронной сети по определению эмоций
        self.__network__.load_model()
        print('[+] Start application')  # отправка в консоль сообщения об успешном старте приложения

    def take_frame(self):
        """
        Метод для обработки изображения с камеры и вывода результата модели, если она смогла предсказать эмоцию
        :return: результирующее изображение после всех обработок
        """
        ret, frame = self.__video_capture__.read()  # загрузка изображения с web-камеры

        if frame is None:
            return None

        if self.__need_flip__:
            frame = cv2.flip(frame, 1)  # отражение изображения относительно вертикальной оси (для зеркального эффекта)

        result = self.__network__.predict(format_image(frame))  # предсказание эмоции по изображению

        # проверка на то, что нейронная сеть смогла предсказать эмоцию по изображению
        if result is not None:
            # создание поля, на котором находятся графики
            cv2.rectangle(frame, (5, 5), (250, 150), (255, 255, 255), -1)

            # отображение предсказания по всем эмоциям
            for index, emotion in enumerate(EMOTIONS):
                # название эмоции
                cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                # диаграмма эмоции (насколько она выражена на изображении)
                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                              (255, 0, 0), -1)

            # отображение наиболее выраженной эмоции
            max_index = np.argmax(result[0])  # индекс самой яркой эмоции
            cv2.rectangle(frame, (251, 5), (251 + 310, 75), COLORS[max_index], -1)
            cv2.putText(frame, EMOTIONS[max_index], (256, 56), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return frame

    def __del__(self):
        """
        Деструктор для закрытия потоков
        """
        self.__video_capture__.release()  # закрытие потока видео с web-камеры
        print('[+] Stop application')  # отправка в консоль сообщения об успешном завершении приложения
        cv2.destroyAllWindows()  # закрытие всех окон

