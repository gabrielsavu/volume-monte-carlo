import cv2
import numpy as np


def load_and_calibrate(filename, dist_coefficients, camera_matrix, preprocessing: bool = True):
    if preprocessing:
        image = preprocessing_image(cv2.imread(f"{filename}.jpg"))
    else:
        image = cv2.imread(f"{filename}.jpg")

    image_points = []
    with open(f"{filename}_image.pts", 'r') as f:
        for line in f.readlines():
            image_points.append(line.split(' '))

    object_points = []
    with open(f"{filename}_object.pts", 'r') as f:
        for line in f.readlines():
            object_points.append(line.split(' '))

    image_points = np.array(image_points, float)
    object_points = np.array(object_points, float)

    _, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coefficients)

    return image, camera_matrix, rotation_vector, translation_vector, dist_coefficients


def preprocessing_image(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    new_image = cv2.GaussianBlur(new_image, (3, 3), 0)
    (_, new_image) = cv2.threshold(new_image, 127, 255, cv2.THRESH_TRIANGLE)
    return undesired_objects(new_image)


def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2


def is_white(image, point):
    real_point = np.array([point], float)
    img_points, _ = cv2.projectPoints(real_point, image[2], image[3], image[1], image[4])

    pixel_x = int(img_points[0][0][0])
    pixel_y = int(img_points[0][0][1])

    return not (image[0])[pixel_x][pixel_y].any()


def calculate_volume(images, n):
    x = np.random.uniform(10, 30, (2, n))
    y = np.random.uniform(10, 30, (2, n))
    z = np.random.uniform(0, 28, (2, n))

    volume_box = 20 * 20 * 28  # cutia are 30 de centimetri (pozele au fost facute in colturile de sus a cutiei)
    print(f"volumul cutiei este de {volume_box} cm^3")

    sigma = 0
    for i in range(0, n):
        is_a_valid_point = 1
        for j in range(0, 2):
            is_a_valid_point *= int(is_white(images[j], np.array([x[j][i], y[j][i], z[j][i]], float)))

        sigma += is_a_valid_point

    volume = (volume_box * sigma / 100) / n  # Pe 100 ca sigma este masurat in pixeli
    print(f"volumul obiectului din cutie este de {volume} cm^3")
    return volume


def draw_domain(image, name):
    real_point = np.array([[0, 0, 0], [40, 0, 0], [40, 0, 28], [0, 0, 28], [0, 40, 0], [40, 40, 0], [0, 40, 28],
                           [40, 40, 28], [20, 20, 14]], float)

    port_img_points, _ = cv2.projectPoints(real_point, image[2], image[3], image[1], image[4])

    cv2.polylines(image[0], [np.array([port_img_points[0][0], port_img_points[1][0], port_img_points[2][0],
                                       port_img_points[3][0]], int)], True, (0, 0, 255), thickness=3)

    cv2.polylines(image[0], [np.array([port_img_points[0][0], port_img_points[1][0], port_img_points[2][0],
                                       port_img_points[3][0]], int)], True, (0, 255, 0), thickness=3)

    cv2.polylines(image[0], [np.array([port_img_points[0][0], port_img_points[1][0], port_img_points[5][0],
                                       port_img_points[4][0]], int)], True, (255, 0, 0), thickness=3)

    cv2.polylines(image[0], [np.array([port_img_points[2][0], port_img_points[3][0], port_img_points[6][0],
                                       port_img_points[7][0]], int)], True, (255, 255, 0), thickness=3)

    cv2.polylines(image[0], [np.array([port_img_points[5][0], port_img_points[4][0], port_img_points[6][0],
                                       port_img_points[7][0]], int)], True, (0, 255, 255), thickness=3)

    cv2.imwrite(f"preprocessed/{name}", image[0])


def main():
    dist_coefficients = np.zeros((5, 1), float)

    camera_matrix = np.array([[5575.0, 0, 3000],
                              [0, 5575.0, 4000],
                              [0, 0, 1]], float)

    # on_top, camera_matrix, rotation_vector, translation_vector, dist_coefficients
    on_top = load_and_calibrate("on_top", dist_coefficients, camera_matrix)
    portrait = load_and_calibrate("portrait", dist_coefficients, camera_matrix)
    images = [on_top, portrait]

    n = 20000
    calculate_volume(images, n)

    cv2.imwrite("preprocessed/on_top.bmp", images[0][0])
    cv2.imwrite("preprocessed/portrait.bmp", images[1][0])

    on_top = load_and_calibrate("on_top", dist_coefficients, camera_matrix, False)
    portrait = load_and_calibrate("portrait", dist_coefficients, camera_matrix, False)

    draw_domain(on_top, "on_top_domain.bmp")
    draw_domain(portrait, "portrait_domain.bmp")


if __name__ == "__main__":
    main()
