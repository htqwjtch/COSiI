import cv2
import numpy as np


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


def apply_dilation(image, ksize1, ksize2):
    kernel = np.ones((ksize1, ksize2), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image


def search_the_contours(image):
    blurred = cv2.GaussianBlur(image, (11, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    return contour_image


def mask_gray(image):
    threshold_value = 35
    _, gray_masked_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return gray_masked_image


def cluster_objects(image, min_corners=4, min_perimeter=22, min_area=1, area_threshold=1000):
    result_image = image.copy()
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    small_contours = []
    areas = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= area_threshold:
            areas.append(area)

    avg_area = np.mean(areas) if areas else 0

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        corner_count = len(approx)
        area = cv2.contourArea(contour)

        if (area < area_threshold or area < avg_area) and peri >= min_perimeter:
            small_contours.append((contour, corner_count))
            continue

        if corner_count >= min_corners and peri >= min_perimeter and area >= min_area:
            results.append({
                'corners': corner_count,
                'perimeter': peri,
                'area': area
            })

            color = get_color_by_corners(corner_count)

            if len(result_image.shape) == 2:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

            cv2.drawContours(result_image, [contour], -1, color, thickness=cv2.FILLED)
        else:
            cv2.drawContours(result_image, [contour], -1, (0, 0, 0), thickness=2)

    if small_contours:
        minimal_corners = calculate_minimal_corners(small_contours)

        color = get_color_by_corners(minimal_corners)

        for contour, _ in small_contours:
            if len(result_image.shape) == 2:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

            cv2.drawContours(result_image, [contour], -1, color, thickness=cv2.FILLED)

    return results, result_image


def calculate_minimal_corners(small_contours):
    total_corners = sum(corner_count for _, corner_count in small_contours)
    # Количество фигур
    p = len(small_contours)
    # Применяем формулу для минимального количества углов
    minimum_corners = total_corners - 2 * (p - 1)
    return minimum_corners


def get_color_by_corners(corner_count):
    if corner_count == 4:
        color = (0, 0, 255)  # Красный
    elif corner_count == 5:
        color = (0, 165, 255)  # Оранжевый
    elif corner_count == 6:
        color = (0, 255, 255)  # Жёлтый
    elif corner_count == 7:
        color = (0, 255, 0)  # Зелёный
    elif corner_count == 8:
        color = (255, 255, 0)  # Голубой
    elif corner_count == 9:
        color = (255, 0, 0)  # Синий
    elif corner_count > 9:
        color = (255, 0, 255)  # Фиолетовый
    else:
        color = (255, 255, 255)  # Белый
    return color


image_path = 'dataset/11.jpg'

if image_path == 'dataset/2.jpg' or image_path == 'dataset/4.jpg' or image_path == 'dataset/6.jpg':
    median_ksize = 5
else:
    median_ksize = 7

image = cv2.imread(image_path)

if image is None:
    print("Ошибка: не удалось загрузить изображение.")
else:
    image = up_scale(image)

    image = mask_blue(image)

    image = apply_Sobel_operator(image, 3)

    image = apply_dilation(image, 3, 3)

    image = apply_median_filter(image, median_ksize)

    image = apply_minimal_filter(image, 3, 3)

    image = mask_gray(image)

    image = search_the_contours(image)

    #cv2.imshow('Contoured_image', image)

    corner_counts, res_image = cluster_objects(image)

    for i, count in enumerate(corner_counts):
        print(f"Объект {i + 1}: {count}")

    cv2.imshow('Colored_image', res_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
