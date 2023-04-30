#
# W_yz2874_zg284 4/28/2023  
#

import numpy as np
import cv2 as cv

# Detect the cue stick and cue ball from a image of the pool table
def detect_cue(image):
    # Mask out everything outside the table with a hsv color scheme
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Dark blue color scheme
    lower = np.array([110, 50, 50])
    upper = np.array([130, 255, 255])
    # Light blue color scheme
    # lower = np.array([90, 50, 50])
    # upper = np.array([120, 255, 255])

    mask = cv.inRange(hsv_img, lower, upper)
    table = cv.bitwise_and(image, image, mask = mask)
    # cv.imshow("cropped table", table)

    # Convert to grayscale and find contours
    table_gray = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(table_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Find the largest contour as the border of the table
    c = max(contours, key = cv.contourArea)
    hull = cv.convexHull(c)
    # hull = cv.minAreaRect(c)
    new_mask = np.zeros_like(image)
    img_new = cv.drawContours(new_mask, [hull], -1, (255, 255, 255), -1)
    # box = cv.boxPoints(hull)
    # box = np.int0(box)
    # img_new = cv.drawContours(new_mask, [box], 0, (255, 255, 255), -1)
    cropped = cv.bitwise_and(image, img_new)
    # cv.imshow("hull", cropped)

    # Detect cue stick by masking pink
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower = np.array([150, 120, 120])
    upper = np.array([165, 255, 255])
    mask = cv.inRange(hsv_img, lower, upper)
    new_img = cv.bitwise_and(image, image, mask = mask)
    # cv.imshow("test", new_img)

    table = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(table, 100, 255, cv.THRESH_BINARY)
    cv.imshow("table", thresh)
    lines = cv.HoughLinesP(thresh, 1, np.pi/180, 40, None, 20, 0)
    print("Line coordinates:", lines)
    cue = 0
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cue += l
            cv.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv.LINE_AA)
        cue = cue / len(lines)
        cue = np.round(cue).astype(int)
    # cv.imshow("images", image)

    # Detect cue ball by masking white
    sensitivity = 80
    hsv_img = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 255 - sensitivity])
    upper = np.array([255, sensitivity, 255])
    mask = cv.inRange(hsv_img, lower, upper)
    new_img = cv.bitwise_and(cropped, cropped, mask = mask)
    # cv.imshow("new img", new_img)

    # Find balls
    # Param1: higher = less circles
    # Param2: higher = less circles
    gray_img = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray_img, cv.HOUGH_GRADIENT, 1, 20, param1=11, param2=11, minRadius=9, maxRadius=15)
    print("Ball coordinates:", circles)
    if lines is not None:
        center = np.array([640, 480])
        d0 = np.linalg.norm(cue[0:2] - center)
        d1 = np.linalg.norm(cue[2:4] - center)
        if d0 < d1:
            cue[0], cue[2] = cue[2], cue[0]
            cue[1], cue[3] = cue[3], cue[1]
        print("Cue stick coordinates:", cue)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            d0 = 1000
            min = 0
            for i in circles[0, :]:
                d1 = np.linalg.norm(i[0:2] - cue[2:4])
                if d1 < d0:
                    d0 = d1
                    min = i
            print("Cue ball coordinates:", min)
            cv.circle(image, min[0:2], min[2], (0, 255, 0), 2)
            cv.circle(image, min[0:2], 2, (0, 0, 255), 3)
            cv.line(image, cue[2:4], min[0:2], (255,255,0), 2, cv.LINE_AA)
            # cv.imshow("images", image)
    # cv.waitKey(0)
    return cue, min, hull[:, 0, :]

# image = cv.imread('./IMG_3686_s.jpg')
# detect_cue(image)