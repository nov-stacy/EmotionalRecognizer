SIZE_FACE = 48  # размер изображения
# список эмоций, которые может определять нейронная сеть
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
# список цветов, ассоциированный с эмоциями
COLORS = [(0, 0, 200), (0, 111, 255), (107, 5, 144), (0, 150, 0), (0, 0, 0), (150, 0, 0), (205, 205, 0)]

SAVE_MODEL_FILENAME = 'network_model'  # название файлов, где хранится нейронная сеть
DATASET_CSV_FILENAME = 'fer2013.csv'  # название файла, где хранится исходный датасет для обучения нейронной сети
SAVE_DATASET_IMAGES_FILENAME = 'data_images.npy'  # название файла, где хранятся признаки датасета после обработки
SAVE_DATASET_LABELS_FILENAME = 'data_labels.npy'  # название файла, где хранятся значения целовой переменной
