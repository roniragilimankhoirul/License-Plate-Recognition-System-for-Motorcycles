import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from database import save_to_database
import base64
import sys

def process_image(image_path):
    # Your existing code for license plate recognition
    img = cv.imread(image_path)
    # Your existing code for license plate recognition
    # img = cv.imread("test images/Plat-H-Semarang.jpg")

    # Resize image by multiplying its size with 0.4
    img = cv.resize(img, (int(img.shape[1] * 0.4), int(img.shape[0] * 0.4)))

    # Convert from BGR to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Create an ellipse-shaped kernel with a diameter of 20 pixels
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

    img_show_plate = img.copy()
    img_show_plate_bw = cv.cvtColor(img_norm_bw, cv.COLOR_GRAY2RGB)

    if len(index_plate_candidate) == 0:
        print("License plate not detected")

    elif len(index_plate_candidate) == 1:
        x_plate, y_plate, w_plate, h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[0]])
        cv.rectangle(img_show_plate, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
        cv.rectangle(img_show_plate_bw, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
        img_plate_gray = img_gray[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]
    else:
        print('Two plate locations found')
        x_plate, y_plate, w_plate, h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[1]])
        cv.rectangle(img_show_plate, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
        cv.rectangle(img_show_plate_bw, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
        img_plate_gray = img_gray[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]

    # Character segmentation using contours
    (thresh, img_plate_bw) = cv.threshold(img_plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    img_plate_bw = cv.morphologyEx(img_plate_bw, cv.MORPH_OPEN, kernel)

    contours_plate, hierarchy = cv.findContours(img_plate_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    index_chars_candidate = []
    index_counter_contour_plate = 0
    img_plate_rgb = cv.cvtColor(img_plate_gray, cv.COLOR_GRAY2BGR)
    img_plate_bw_rgb = cv.cvtColor(img_plate_bw, cv.COLOR_GRAY2RGB)

    for contour_plate in contours_plate:
        x_char, y_char, w_char, h_char = cv.boundingRect(contour_plate)
        if h_char >= 40 and h_char <= 60 and w_char >= 10:
            index_chars_candidate.append(index_counter_contour_plate)
            cv.rectangle(img_plate_rgb, (x_char, y_char), (x_char + w_char, y_char + h_char), (0, 255, 0), 5)
            cv.rectangle(img_plate_bw_rgb, (x_char, y_char), (x_char + w_char, y_char + h_char), (0, 255, 0), 5)
        index_counter_contour_plate += 1

    # Character sorting
    x_coors = []

    for char in index_chars_candidate:
        x, y, w, h = cv.boundingRect(contours_plate[char])
        x_coors.append(x)

    x_coors = sorted(x_coors)
    index_chars_sorted = []

    for x_coor in x_coors:
        for char in index_chars_candidate:
            x, y, w, h = cv.boundingRect(contours_plate[char])
            if x_coors[x_coors.index(x_coor)] == x:
                index_chars_sorted.append(char)

    # Character classification using a trained model
    img_height = 40
    img_width = 40
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    model = keras.models.load_model('model')

    num_plate = []

    for char_sorted in index_chars_sorted:
        x, y, w, h = cv.boundingRect(contours_plate[char_sorted])
        char_crop = cv.cvtColor(img_plate_bw[y:y + h, x:x + w], cv.COLOR_GRAY2BGR)
        char_crop = cv.resize(char_crop, (img_width, img_height))
        img_array = keras.preprocessing.image.img_to_array(char_crop)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        num_plate.append(class_names[np.argmax(score)])
        print(class_names[np.argmax(score)], end='')

    plate_number = ''
    for a in num_plate:
        plate_number += a

    cv.putText(img_show_plate, plate_number, (x_plate, y_plate + h_plate + 50), cv.FONT_ITALIC, 2.0, (0, 255, 0), 3)
    # Assuming you have extracted the plate number into the variable 'plate_number'
    # and the final image is in the variable 'img_show_plate'

    # Display the output
    cv.imshow("License Plate Recognition", img_show_plate)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Convert the image to base64
    _, img_encode = cv.imencode('.jpg', img_show_plate)
    img_base64 = base64.b64encode(img_encode).decode('utf-8')

    # Save the result to the database
    save_to_database(plate_number, img_base64)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    process_image(image_path)