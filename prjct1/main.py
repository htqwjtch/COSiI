import cv2
import numpy as np


# Абсолютный путь к изображению
image_path = 'dataset/5.jpg'

# Чтение изображения
image = cv2.imread(image_path)

# Проверка, что изображение было успешно загружено
if image is None:
    print("Ошибка: не удалось загрузить изображение.")
else:
    if image_path == 'dataset/11.jpg':
        scale_percent = 40

    else:
        scale_percent = 20

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Преобразование изображения в цветовое пространство HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Определение нижнего и верхнего порогов синего цвета
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Создание маски для синего цвета
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Применение маски к изображению
    result = cv2.bitwise_and(image, image, mask=mask)

    gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    gray_image = (gray_image * 0.04).astype(np.uint8)

    # # Определение ядра (структурного элемента) для фильтра
    kernel_min = np.ones((5, 5), np.uint8)

    # Применение минимального фильтра с помощью операции эрозии
    gray_image = cv2.erode(gray_image, kernel_min)

    gray_image = (gray_image * 3).astype(np.uint8)

    # Применение оператора Собеля
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # По оси X
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # По оси Y

    # Комбинирование результатов
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Преобразование результата в 8-битное изображение для отображения
    sobel_combined = cv2.convertScaleAbs(sobel_combined)

    # Отображение результата
    cv2.imshow('Original Image', image)
    cv2.imshow('Sobel Combined', sobel_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
