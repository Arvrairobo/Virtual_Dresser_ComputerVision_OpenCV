# import the necessary packages
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy as np

def align_helper(f, image, gray, rect):

    shape = f.predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - f.desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - f.desiredLeftEye[0])
    desiredDist *= f.desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
        (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = f.desiredFaceWidth * 0.5
    tY = f.desiredFaceHeight * f.desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (f.desiredFaceWidth, f.desiredFaceHeight)
    output = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC)

    # return the aligned face
    return (output, angle)
 
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# help="path to input image")
# args = vars(ap.parse_args())


# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])

# # INPUT DESIRED WIDTH OF IMAGE 
# fa = face_utils.FaceAligner(predictor, desiredFaceWidth=500)

# image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# rects = detector(gray, 2)

# # loop over the face detections
# for rect in rects:

#     (x, y, w, h) = face_utils.rect_to_bb(rect)
#     faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
    
#     (faceAligned, align_angle) = align_helper(fa, image, gray, rect)
#     print ("Align angle in degrees", align_angle)

#     # CAN ALSO SAVE, AS PER NEEDED
#     cv2.imshow("Aligned", faceAligned)
#     cv2.waitKey(0)