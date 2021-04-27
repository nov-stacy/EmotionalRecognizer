from config.constants import *
from network.emotion_recognition import EmotionRecognition
from network.dataset_loader import DatasetLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_plot_matrix(matrix_value, label, filename, mask, title_plot, rotation_y=90, rotation_x=0):
    """
    Метод для создания тепловой карты для матрицы
    :param matrix_value: значение матрицы
    :param label: значения для осей
    :param filename: имя файла для сохранения
    :param mask: маска, по которой следует отображать матрицу (возможно удаление некоторых значений)
    :param title_plot: название графика
    :param rotation_y: поворот значений оси Y
    :param rotation_x: поворот значений оси X
    """
    # матрица с вещественными значениями сокращенными до сотых
    new_matrix = [[float("%.2f" % value) for value in line] for line in matrix_value]
    ax = sns.heatmap(new_matrix, annot=True, cmap="YlGnBu", mask=mask)  # тепловая карта
    plt.title(title_plot)  # название графика
    # значения осей
    ax.set_xticklabels(label, minor=False, rotation=rotation_x)
    ax.set_yticklabels(label, minor=False, rotation=rotation_y)
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Real Emotion')
    # сохранение
    plt.tight_layout()
    plt.savefig('results/' + filename, dpi=plt.gcf().dpi)
    plt.show()


def matrix_class(data_value, index_value):
    """
    Метод для генерации матрицы ошибок для одного класса
    :param data_value: матрица, из которой берутся значения
    :param index_value: индекс эмоции
    :return: матрица ошибок
    """
    tp = data_value[index_value, index_value]
    fn = sum(data_value[index_value, 0: index_value]) + sum(data_value[index_value, index_value + 1: len(EMOTIONS)])
    fp = sum(data_value[0: index_value, index_value]) + sum(data_value[index_value + 1: len(EMOTIONS), index_value])
    tp, fn, fp = tp / (tp + fn + fp), fn / (tp + fn + fp), fp / (tp + fn + fp)
    tn = 0

    # сохранение результатов
    with open('results/' + EMOTIONS[index_value] + '/values.txt', 'w') as file_values:
        print('precision: ', tp / (fp + tp), file=file_values)
        print('recall: ', tp / (tp + fn), file=file_values)

    return np.array([[tp, fn], [fp, tn]])


if __name__ == '__main__':

    # загрузка нейронной сети для определения эмоций человека
    network = EmotionRecognition()
    network.load_model(directory='../network/')

    # загрузка целевых значений и результатов работы алгоритма
    DatasetLoader.load()
    labels = DatasetLoader.labels()
    with open('../data/result.txt') as file:
        result = [list(map(float, line.split())) for line in file]

    print('[+] Loading Data')  # отправка в консоль сообщения об успешном начале работы

    data = np.zeros((len(EMOTIONS), len(EMOTIONS)))  # таблица ошибок и верных ответов для всех эмоций

    # для каждого значения из выборки происходит предсказание, после чего оно добавляется в матрицу ошибок
    for index in range(len(result)):
        data[np.argmax(labels[index]), result[index].index(max(result[index]))] += 1  # матрица ошибок

    print('[+] Create Data Matrix')  # отправка сообщения в консоль для отсечения стадии обработки

    # находит процент от колонки
    # (для того чтобы определить отношение верных ответов относительно всех истинных значений)
    data1 = (data.T / np.sum(data, axis=1)).T
    data2 = data / len(labels)

    print('[+] Create Data Matrix 1 and 2')  # отправка сообщения в консоль для отсечения стадии обработки

    mask_emotions = [[False] * len(EMOTIONS)] * len(EMOTIONS)  # маска для отображения всех матрицы
    generate_plot_matrix(data, EMOTIONS, 'matrix_data.svg', mask_emotions,
                         'Value matrix (data)', rotation_y=0, rotation_x=30)
    generate_plot_matrix(data1, EMOTIONS, 'matrix_columns.svg', mask_emotions,
                         'Value matrix (by columns)', rotation_y=0, rotation_x=30)
    generate_plot_matrix(data2, EMOTIONS, 'matrix_general.svg', mask_emotions,
                         'Value matrix (general)', rotation_y=0, rotation_x=30)

    print('[+] Generating Graph Matrix')  # отправка сообщения в консоль для отсечения стадии обработки

    # значение accuracy для модели
    with open('results/accuracy.txt', 'w') as file:
        print(sum([data[index][index] for index in range(len(EMOTIONS))]) / len(labels), file=file)

    print('[+] Generating Accuracy')  # отправка сообщения в консоль для отсечения стадии обработки

    # генерация графиков для всех классов
    for index in range(len(EMOTIONS)):
        matrix = matrix_class(data, index)
        mask_value = [[False, False], [False, True]]  # маска для отображения всей матрицы кроме tn
        generate_plot_matrix(matrix, [1, 0], EMOTIONS[index] + '/matrix.svg', mask_value, EMOTIONS[index].title())

    print('[+] Generating Class Graphs')  # отправка сообщения в консоль для отсечения стадии обработки
