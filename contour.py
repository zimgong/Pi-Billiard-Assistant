import numpy as np
import argparse
import cv2 as cv

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "Path to the image")
# args = vars(ap.parse_args())

# image = cv.imread(args["image"])

# image = cv.imread('./blue_table.png')
image = cv.imread('./IMG_3602_s.JPG')
# image = cv.imread('./IMG_3600_s.JPG')

# Hard code color range for table and mask out everything else
# Should work for blue or relevant colored table

# These are perfect for light blue! 
lower = 100, 100, 0
upper = 240, 160, 120

# These are for dark blue! 
# lower = 80, 80, 0
# upper = 240, 150, 120

# Mask out everything outside the table
mask = cv.inRange(image, lower, upper)
table = cv.bitwise_and(image, image, mask = mask)
# cv.imshow("images", table)

# Convert to grayscale and find contours
table_gray = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
contours, hierarchy = cv.findContours(table_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# Find the largest contour as the border of the table
c = max(contours, key = cv.contourArea)
hull = cv.convexHull(c)
new_mask = np.zeros_like(image)
img_new = cv.drawContours(new_mask, [hull], -1, (255, 255, 255), -1)
cropped = cv.bitwise_and(image, img_new)
# cv.imshow("images", cropped)

# Mask the table out
balls = cv.bitwise_and(cropped, cropped, mask = 255-mask)
balls = cv.cvtColor(balls, cv.COLOR_BGR2GRAY)
# cv.imshow("images", balls)
balls = cv.medianBlur(balls, 5)
# cv.imshow("images", balls)

# Find balls
# Param1: higher = less circles
# Param2: higher = less circles
contours, hierarchy = cv.findContours(balls, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
c = max(contours, key = cv.contourArea)
img_new = cv.drawContours(image, c, -1, (255, 255, 255), 2)

# for i in circles[0, :]:
#     cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# cv.imshow("images", np.hstack([cropped, image]))
cv.imshow("images", image)

cv.waitKey(0)