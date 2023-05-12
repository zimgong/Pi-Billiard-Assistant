#
# W_yz2874_zg284 5/9/2023 Pi Camera Picture Test
#

import cv2 as cv

RES_X = 640
RES_Y = 480
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, RES_X)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, RES_Y)

ret, frame = cap.read()

cv.imwrite('frame.jpg', frame)

cap.release()