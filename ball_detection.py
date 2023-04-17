import numpy as np
import argparse
import cv2 as cv

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "Path to the image")
# args = vars(ap.parse_args())

# image = cv.imread(args["image"])

# image = cv.imread('./blue_table.png')
image = cv.imread('./IMG_3602_s.JPG')

# Hard code color range for table and mask out everything else
# Should work for blue or relevant colored table
lower = 100, 100, 0
upper = 240, 140, 120
# Mask out everything outside the table
mask = cv.inRange(image, lower, upper)
table = cv.bitwise_and(image, image, mask = mask)

# Convert to grayscale and find contours
table_gray = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
contours, hierarchy = cv.findContours(table_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# Find the largest contour as the border of the table
c = max(contours, key = cv.contourArea)
hull = cv.convexHull(c)
new_mask = np.zeros_like(image)
img_new = cv.drawContours(new_mask, [hull], -1, (255, 255, 255), -1)
cropped = cv.bitwise_and(image, img_new)

# Mask the table out
balls = cv.bitwise_and(cropped, cropped, mask = 255-mask)
balls = cv.cvtColor(balls, cv.COLOR_BGR2GRAY)
balls = cv.medianBlur(balls, 5)

# Find balls
# Param2: higher = less circles
circles = cv.HoughCircles(balls, cv.HOUGH_GRADIENT, 1, 20, param1=10, param2=11, minRadius=7, maxRadius=12)
circles = np.uint16(np.around(circles))
# print(circles)
# print(circles.shape)
for i in circles[0, :]:
    cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# cv.imshow("images", np.hstack([table, cropped]))
cv.imshow("images", image)

cv.waitKey(0)