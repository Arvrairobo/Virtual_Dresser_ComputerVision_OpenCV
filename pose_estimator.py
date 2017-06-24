import cv2
import numpy as np
import argparse
from imutils import face_utils
import imutils
import dlib
import math

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#     help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
#     help="path to input image")

# args = vars(ap.parse_args())

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])

# image = cv2.imread(args["image"])


nose_tip = 0 # 34
chin = 0 # 9
leyelcorner = 0 # 46
reyercorner = 0 # 18
lmouthcorner = 0 # 55
rmouthcorner = 0 # 49


def getPoints(im, detector, predictor):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    size = im.shape
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
      
        count = 0
        for (x,y) in shape :
            #cv2.circle(im, (x, y), 1, (0, 0, 255), -1)
            if count == 34 :
                nose_tip = (x,y)
            if count == 9 :
                chin = (x,y)
            if count == 46 :
                leyelcorner = (x,y)
            if count == 18 :
                reyercorner = (x,y)
            if count == 55 :
                lmouthcorner = (x,y)
            if count == 49 :
                rmouthcorner = (x,y)
            count += 1

    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                nose_tip,     # Nose tip
                                chin,     # Chin
                                leyelcorner,     # Left eye left corner
                                reyercorner,     # Right eye right corne
                                lmouthcorner,     # Left Mouth corner
                                rmouthcorner      # Right mouth corner
                            ], dtype="double")
     
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                             
                            ])

    # Camera internals
     
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
     
     
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs) 
     
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
     
    # cv2.line(im, p1, p2, (255,0,0), 2)
    # cv2.imshow("line", im)
    # cv2.waitKey(0)
    sign = 0

    if p2[0] >= 0 :
        sign = -1
    else : 
        sign = 1

    return (p1, p2, sign)

def getAngle(image, detector, predictor):

    # angle is neg if tilted to left
    (pa_a, pa_b, sign) = getPoints(image, detector, predictor)

    slope_pa = (pa_b[1] - pa_a[1]) / float((pa_b[0] - pa_a[0]))

    r = 180 / 3.14
    angle_pa = (math.atan(1 / slope_pa)) * r * sign

    # print ("angle", angle_pa, "sign", sign)
    return (angle_pa)

#print(getAngle(image, detector, predictor))
