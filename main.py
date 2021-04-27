import tkinter
import cv2
from PIL import Image, ImageTk
from camera import Camera
import os


camera, label_main = None, None


def turn_camera():
    """
    Метод для изменения состояния камеры (включена или выключена)
    """
    global is_used_camera
    is_used_camera = not is_used_camera  # изменение состояния камеры на противположное
    show_frame()  # продолжить демонстрацию изображения


def show_frame():
    """
    Метод для показа изображения
    """
    global is_used_camera, camera
    # проверка на работу камеры (если не работает, то изображения не будет)
    if not is_used_camera:
        return

    # получение изображения с результатам работы всех систем с камеры и его постобработка
    frame = camera.take_frame()
    if frame is None:
        del camera
        exit(0)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img_tk = ImageTk.PhotoImage(image=img)
    label_main.imgtk = img_tk
    label_main.configure(image=img_tk)
    label_main.after(10, show_frame)


if __name__ == '__main__':

    print('If you want to use video from your device write relative path\n'
          'Otherwise the application will be opened with web-camera')

    path = input()

    if path.endswith('.avi') or path.endswith('.mp4'):
        if os.path.exists(path):
            camera = Camera(path)
        else:
            print('[-] There is no such file or file is incorrect')
            exit(0)
    else:
        camera = Camera()

    is_used_camera = True  # камера включена

    # создание главного экрана
    window = tkinter.Tk()
    window.title('The Program for Dynamic Assessment of Emotional Facial Expressions of a Person')
    window['bg'] = "#555"

    # создание окна для демонстрации изображений с камеры
    image_frame = tkinter.Frame(window, width=600, height=500, background="#555")
    image_frame.grid(row=0, column=0, padx=10, pady=10)
    label_main = tkinter.Label(image_frame)
    label_main.grid()

    # создание кнопки
    btn = tkinter.Button(window, text="Turn the camera on/off", command=turn_camera, width=143, foreground="#ccc",
                         highlightbackground="#555", fg="Black", highlightthickness=10)
    btn.grid()

    # начало работы
    show_frame()
    window.mainloop()

    # удаление камеры
    del camera
