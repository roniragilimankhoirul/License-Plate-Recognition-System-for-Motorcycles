import cv2 as cv
import numpy as np
from tensorflow import keras
import tensorflow as tf

def process_image(image_path):
    img = cv.imread(image_path)
    img = cv.resize(img, (int(img.shape[1] * 0.4), int(img.shape[0] * 0.4)))
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Return only the processed grayscale image
    return img_gray


def detect_license_plate(img_gray):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))

    # Light normalization (1)
    img_opening = cv.morphologyEx(img_gray, cv.MORPH_OPEN, kernel)

    # Light normalization (2)
    img_norm = img_gray - img_opening

    # Light normalization (3)
    _, img_norm_bw = cv.threshold(img_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Plate detection using contours
    contours_vehicle, hierarchy = cv.findContours(img_norm_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    index_plate_candidate = []
    index_counter_contour_vehicle = 0

    for contour_vehicle in contours_vehicle:
        x, y, w, h = cv.boundingRect(contour_vehicle)
        aspect_ratio = w / h
        if w >= 200 and aspect_ratio <= 4:
            index_plate_candidate.append(index_counter_contour_vehicle)
        index_counter_contour_vehicle += 1

    if len(index_plate_candidate) == 0:
        print("License plate not detected")
        return None
    elif len(index_plate_candidate) == 1:
        x_plate, y_plate, w_plate, h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[0]])
    else:
        print('Two plate locations found')
        x_plate, y_plate, w_plate, h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[1]])

    img_show_plate = img.copy()
    img_show_plate_bw = cv.cvtColor(img_norm_bw, cv.COLOR_GRAY2RGB)

    cv.rectangle(img_show_plate, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
    cv.rectangle(img_show_plate_bw, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
    img_plate_gray = img_gray[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]

    return {'img_show_plate': img_show_plate, 'img_show_plate_bw': img_show_plate_bw, 'img_plate_gray': img_plate_gray}

def segment_characters(img_plate_gray):
    # Character segmentation using contours
    (thresh, img_plate_bw) = cv.threshold(img_plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    img_plate_bw = cv.morphologyEx(img_plate_bw, cv.MORPH_OPEN, kernel)

    contours_plate, _ = cv.findContours(img_plate_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    index_chars_candidate = []
    index_counter_contour_plate = 0
    img_plate_rgb = cv.cvtColor(img_plate_gray, cv.COLOR_GRAY2BGR)
    img_plate_bw_rgb = cv.cvtColor(img_plate_bw, cv.COLOR_GRAY2RGB)

    for contour_plate in contours_plate:
        x_char, y_char, w_char, h_char = cv.boundingRect(contour_plate)
        if 40 <= h_char <= 60 and w_char >= 10:
            index_chars_candidate.append(index_counter_contour_plate)
            cv.rectangle(img_plate_rgb, (x_char, y_char), (x_char + w_char, y_char + h_char), (0, 255, 0), 5)
            cv.rectangle(img_plate_bw_rgb, (x_char, y_char), (x_char + w_char, y_char + h_char), (0, 255, 0), 5)
        index_counter_contour_plate += 1

    # Character sorting
    x_coors = [cv.boundingRect(contours_plate[char])[0] for char in index_chars_candidate]
    x_coors = sorted(x_coors)
    index_chars_sorted = [char for x_coor in x_coors for char in index_chars_candidate if
                          x_coors[x_coors.index(x_coor)] == cv.boundingRect(contours_plate[char])[0]]

    return {'img_plate_rgb': img_plate_rgb, 'img_plate_bw_rgb': img_plate_bw_rgb, 'index_chars_sorted': index_chars_sorted}

def classify_characters(img_plate_bw, contours_plate, model, class_names):
    # Character classification using a trained model
    img_height = 40
    img_width = 40

    num_plate = []
    for char_sorted in contours_plate['index_chars_sorted']:
        x, y, w, h = cv.boundingRect(contours_plate['contours_plate'][char_sorted])
        char_crop = cv.cvtColor(img_plate_bw[y:y + h, x:x + w], cv.COLOR_GRAY2BGR)
        char_crop = cv.resize(char_crop, (img_width, img_height))
        img_array = keras.preprocessing.image.img_to_array(char_crop)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        num_plate.append(class_names[np.argmax(score)])
        print(class_names[np.argmax(score)], end='')

    plate_number = ''.join(num_plate)

    cv.putText(contours_plate['img_plate_rgb'], plate_number, (cv.boundingRect(contours_plate['contours_plate'][contours_plate['index_chars_sorted'][0]])[0], 
                                                               cv.boundingRect(contours_plate['contours_plate'][contours_plate['index_chars_sorted'][0]])[1] + 
                                                               cv.boundingRect(contours_plate['contours_plate'][contours_plate['index_chars_sorted'][0]])[3] + 50),
               cv.FONT_ITALIC, 2.0, (0, 255, 0), 3)

    return {'plate_number': plate_number, 'img_with_plate_number': contours_plate['img_plate_rgb']}
