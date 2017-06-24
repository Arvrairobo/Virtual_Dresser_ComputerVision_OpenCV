from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import superimpose_api
import math

#####################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-j", "--image_necklace", required = True, 
        help = "path to earring image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#####################################################################

image = cv2.imread(args["image"])
image_necklace = cv2.imread(args["image_necklace"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

#####################################################################

x_offset = []
y_offset = []

#####################################################################

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
  
    (start, end) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
    count = 0
    for (x,y) in shape[start : end] :
        if count == 5 :
            x_offset.append(x)
            y_offset.append(y)
        count += 1

######################################################################### 

angle = 0
bodyy = y_offset[0]
bodyx = x_offset[0]

# will be ratio wrt to distance from cam
offset = 25

Superimposition = superimpose_api.Superimposition()

print(bodyx, bodyy)

image = cv2.cvtColor(Superimposition.superimpose(image_necklace, image, 0, bodyx - offset, bodyy, 20, "necklace"), cv2.COLOR_BGR2RGB)

cv2.imshow("output", image)
cv2.waitKey(0)

#####################################################################
