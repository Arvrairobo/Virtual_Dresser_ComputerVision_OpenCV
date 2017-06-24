# python superimpose_earring_bind.py --shape-predictor shape_predictor_68_face_landmarks.dat --image Pics/2.png --image_earring Pics/earrings2.jpg 


from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import superimpose_api
import math
import pose_estimator

#####################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-j", "--image_earring", required = True, 
        help = "path to earring image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#####################################################################

image = cv2.imread(args["image"])
image_earring = cv2.imread(args["image_earring"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)


x_offset = []
y_offset = []

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (start, end) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
    count = 0

    for (x,y) in shape[start : end] :

        if count == 2 or count == 14 :
            x_offset.append(x)
            y_offset.append(y)

        count += 1

ear_lobe_offset_x = 10 # MAGIC NUMBER

#####################################################################

angle = pose_estimator.getAngle(image, detector, predictor)
bodyy = y_offset[0] + ear_lobe_offset_x
bodyx = x_offset[0] - ear_lobe_offset_x * 3
bodyyy = y_offset[1] + ear_lobe_offset_x
bodyxx = x_offset[1] + ear_lobe_offset_x / 2

Superimposition = superimpose_api.Superimposition()


print("angle", angle)


# ANGLE TO BE SET 
# @TODO
# if right ear can be seen 
if angle > -10 :
    image = cv2.cvtColor(Superimposition.superimpose(image_earring, image, angle, bodyx, bodyy, 3, "earring"), cv2.COLOR_BGR2RGB)

# if left ear can be seen 
if angle < 10 :
    image = cv2.cvtColor(Superimposition.superimpose(image_earring, image, angle, bodyxx, bodyyy,3, "earring"), cv2.COLOR_BGR2RGB)

resized = cv2.resize(image, (int(image.shape[1] / 2.5), int(image.shape[0] / 2.5)))
cv2.imshow("output", resized)
cv2.waitKey(0)

#####################################################################


