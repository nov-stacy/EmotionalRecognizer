from network.emotion_recognition import EmotionRecognition
from network.dataset_loader import DatasetLoader


if __name__ == '__main__':

    # загрузка нейронной сети для определения эмоций человека
    network = EmotionRecognition()
    network.load_model(directory='../network/')

    # загрузка всего датасета
    images, labels = DatasetLoader.images(), DatasetLoader.labels()

    print('[+] Loading Data')  # отправка в консоль сообщения об успешном начале работы

    with open('../data/result.txt', 'w') as file:
        # для каждого значения из выборки происходит предсказание
        for index in range(images.shape[0]):
            # отправка сообщения в консоль для определения стадии обработки
            print('It was the image number: ', index, '/', images.shape[0] - 1)
            result = network.predict(images[index])[0]  # результат работы нейронной сети
            print(' '.join([str(res) for res in result]), file=file)
