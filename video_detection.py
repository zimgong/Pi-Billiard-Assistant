import numpy as np
import argparse
import cv2 as cv

# cap = cv.VideoCapture('./3076_720p.mp4')
cap = cv.VideoCapture('./696_720p.mp4')

lower = 100, 100, 0
upper = 240, 140, 120
first = True

while cap.isOpened():
    ret, frame = cap.read()
    
    if first:
    # Hard code color range for table and mask out everything else
        mask = cv.inRange(frame, lower, upper)
        table = cv.bitwise_and(frame, frame, mask = mask)

        table_gray = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(table_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Find the largest contour as the border of the table
        c = max(contours, key = cv.contourArea)
        hull = cv.convexHull(c)
        new_mask = np.zeros_like(frame)
        img_new = cv.drawContours(new_mask, [hull], -1, (255, 255, 255), -1)
        first = False

    cropped = cv.bitwise_and(frame, img_new)

    # Mask the table out
    balls = cv.bitwise_and(cropped, cropped, mask = 255-mask)
    balls = cv.cvtColor(balls, cv.COLOR_BGR2GRAY)
    balls = cv.medianBlur(balls, 5)

    # Find balls
    # Param2: higher = less circles
    circles = cv.HoughCircles(balls, cv.HOUGH_GRADIENT, 1, 20, param1=10, param2=12, minRadius=7, maxRadius=12)
    if circles.any():
        circles = np.uint16(np.around(circles))
        # print(circles)
        # print(circles.shape)
        for i in circles[0, :]:
            cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    # cv.imshow("images", np.hstack([image, output]))
    cv.imshow("frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()