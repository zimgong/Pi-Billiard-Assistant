import numpy as np
import argparse
import cv2 as cv

# cap = cv.VideoCapture('./3076_720p.mp4')
# lower = 80, 80, 0
# upper = 240, 150, 120

# cap = cv.VideoCapture('./696_720p.mp4')
# cap = cv.VideoCapture('./716_720p.mp4')
cap = cv.VideoCapture('./717_720p.mp4')


# Retake table map every 120 frames to avoid effect of hand movement
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if count == 0:
        # Hard code color range for table and mask out everything else
        hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower = np.array([90, 50, 50])
        upper = np.array([120, 255, 255])
        mask = cv.inRange(hsv_img, lower, upper)
        table = cv.bitwise_and(frame, frame, mask = mask)

        table_gray = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(table_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Find the largest contour as the border of the table
        c = max(contours, key = cv.contourArea)
        hull = cv.convexHull(c)
        new_mask = np.zeros_like(frame)
        img_new = cv.drawContours(new_mask, [hull], -1, (255, 255, 255), -1)

    cropped = cv.bitwise_and(frame, img_new)

    sensitivity = 70
    hsv_img = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 255 - sensitivity])
    upper = np.array([255, sensitivity, 255])
    mask = cv.inRange(hsv_img, lower, upper)
    new_img = cv.bitwise_and(cropped, cropped, mask = mask)

    # Find balls
    # Param2: higher = less circles
    gray_img = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray_img, cv.HOUGH_GRADIENT, 1, 20, param1=10, param2=10, minRadius=7, maxRadius=13)
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

    count += 1
    if count >= 30:
        count = 0

cap.release()
cv.destroyAllWindows()