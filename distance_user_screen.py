# getDistance(image_calibrate, image)
# image_calibrate is at Pics/distance_to_camera_2ft.jpg

import numpy as np
import cv2
import argparse
import dlib
from imutils import face_utils
import imutils

#####################################################################

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-j", "--image_calibrate", required=True,
    help="path to input image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


#####################################################################

def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)

    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

def getDistance(image_calibrate, image):

    # image_calibrate is at Pics/distance_to_camera_2ft.jpg
    # distance is 2ft or 24 inches 
    # width is 11 inches

    KNOWN_DISTANCE = 24.0
    KNOWN_WIDTH = 11.0

    marker = find_marker(image_calibrate)
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    face_coords = []

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
      
        (start, end) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

        count = 0

        for (x,y) in shape[start : end] :
            
            if count == 0 or count == 16 :
                face_coords.append(x)
            count += 1

    # multiplying factor is using DPI of screen 
    # https://www.ninjaunits.com/converters/pixels/pixels-inches/#dpi-finder

    KNOWN_WIDTH_NEW = (face_coords[1] - face_coords[0] + 10)*0.0133

    marker_new = find_marker(image)
    inches = distance_to_camera(KNOWN_WIDTH_NEW, focalLength, marker[1][0])
    return inches

#####################################################################

# UNCOMMENT TO TEST
# image_calibrate = cv2.imread(args["image_calibrate"])
# image = cv2.imread(args["image"])
#print(getDistance(image_calibrate, image))

#####################################################################
