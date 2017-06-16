# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from align_faces_earrings import align_helper

#####################################################################

# Get all the arguments, initialize face detector and predictor

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-j", "--image_earring", required = True, 
        help = "path to earring image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#####################################################################

# load the input image, resize it, and convert it to grayscale

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 2)

#####################################################################

# RESIZE IMAGE HERE
# ERROR if image of earring has >= size than actual image

image_earring = cv2.imread(args["image_earring"], 1)
image_earring = imutils.resize(image_earring, width = 100) # random resize

#####################################################################

# positions of superimposing image
x_offset = []
y_offset = []

#####################################################################

# align faces
# WILL NOT ALIGN FACE - WILL USE ANGLE TO CHANGE EARRING ORIENTATION
# @TODO

# fa = face_utils.FaceAligner(predictor, desiredFaceWidth=500)
# for rect in rects:

#     (x, y, w, h) = face_utils.rect_to_bb(rect)
#     (image , align_angle) = align_helper(fa, image, gray, rect)
#     print ("Align angle in degrees", align_angle)

######################################################################
 
# detect faces in the grayscale image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
 
#######################################################################
    
    # Prints out which facial landmark, index, coordinates
    # UNCOMMENT FOR DEBUGGING
    # for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
    #     for (idx, (x,y)) in enumerate(shape[i:j]):
    #         print(name, (i,j),"idx num", idx + i ,  "coords", (x,y))
    
########################################################################
  
    # draws yellow points for jaw landmarks
    (start, end) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
    count = 0
    # (x,y) are the coordinates
    for (x,y) in shape[start : end] :
        # UNCOMMENT FOR DEBUGGING
        # cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
        # cv2.putText(image, str(count), (x,y),cv2.FONT_HERSHEY_SIMPLEX,
        #                               0.4, (255, 100, 0), 1)
        if count == 2 or count == 14 :
            x_offset.append(x)
            y_offset.append(y)

        count += 1

#########################################################################
# Super impose at particular coordinates got from above, for earrings the 
# index 2 and 14 are close to the earlobe 

# ASSUMPTION : earring is in center of image
semi_width = image_earring.shape[1] / 2 
ear_lobe_offset_x = 10 # MAGIC NUMBER

count_image_offset_x = 0
for col in range(x_offset[i] + image_earring.shape[1] - x_offset[i]):
    if sum(image_earring[semi_width, col]) == 0: 
        count_image_offset_x += 1
    else : break

count_image_offset_y = 0
for row in range(y_offset[i] + image_earring.shape[0] - y_offset[i]):
    if sum(image_earring[row, semi_width] == 0):
        count_image_offset_y += 1
    else : break

# GETS RID OF BACKGROUND
for i in range(2):
    ystart = y_offset[i]
    yend = y_offset[i] + image_earring.shape[0]
    xstart = x_offset[i]
    xend = x_offset[i] + image_earring.shape[1]

    y_range = abs(ystart - yend)
    x_range = abs(xstart - xend)

    for row in range(y_range):
        for col in range(x_range):
            if sum(image_earring[row, col] == 0) : continue
            else :
                if i == 0: 
                    image[ystart + row - count_image_offset_y, xstart + col - count_image_offset_x - ear_lobe_offset_x] = image_earring[row, col]
                else :
                    image[ystart + row - count_image_offset_y, xstart + col - count_image_offset_x + 0*ear_lobe_offset_x] = image_earring[row, col]
     
#########################################################################


#########################################################################


# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)