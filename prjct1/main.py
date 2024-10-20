import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def up_scale(image):
    scale_percent = 20 if image_path != 'dataset/11.jpg' else 40
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


def mask_blue(image):
    lower_blue = np.array([100, 150, 100]) if image_path == 'dataset/11.jpg' else np.array([100, 150, 200])
    upper_blue = np.array([140, 255, 255])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_masked_image = cv2.bitwise_and(image, image, mask=mask)
    return blue_masked_image


def apply_minimal_filter(image, ksize1, ksize2):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    kernel_min = np.ones((ksize1, ksize2), np.uint8)
    filtered_image = cv2.erode(gray_image, kernel_min)
    return filtered_image


def apply_median_filter(image, ksize):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    filtered_image = cv2.medianBlur(gray_image, ksize)
    return filtered_image


def apply_Sobel_operator(image, ksize):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_image = cv2.convertScaleAbs(sobel_combined)
    return sobel_image


def apply_Laplass_filter(image):
    high_freq_filter = cv2.Laplacian(image, cv2.CV_64F)
    laplass_image = np.uint8(np.absolute(high_freq_filter))
    return laplass_image


def apply_dilation(image, ksize1, ksize2):
    kernel = np.ones((ksize1, ksize2), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image


def search_the_contours(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    return contour_image


def mask_gray(image):
    threshold_value = 20
    _, gray_masked_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return gray_masked_image


def count_corners(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corner_counts = []
    for contour in contours:
        # Периметр контура
        peri = cv2.arcLength(contour, True)
        # Аппроксимация контура многоугольником
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        # Количество углов — это количество вершин аппроксимированного многоугольника
        corner_counts.append(len(approx))

    return corner_counts


image_path = 'dataset/11.jpg'

image = cv2.imread(image_path)

if image is None:
    print("Ошибка: не удалось загрузить изображение.")
else:
    image = up_scale(image)

    image = mask_blue(image)

    # image = apply_median_filter(image, 3)

    image = apply_Sobel_operator(image, 3)

    #image = apply_Laplass_filter(image)

    image = apply_dilation(image, 3, 3)

    #image = apply_minimal_filter(image, 3, 1)

    image = apply_median_filter(image, 3)

    image = mask_gray(image)

    image = search_the_contours(image)

    cv2.imshow('Contoured_image', image)

    corner_counts = count_corners(image)

    for i, count in enumerate(corner_counts):
        print(f"Объект {i + 1}: {count} углов.")

    # cv2.imshow('Colored_image', output_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
