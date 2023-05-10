#
# W_yz2874_zg284 4/28/2023 Frame for Ball Detection
#

import numpy as np
import cv2 as cv

# Load image
# image = cv.imread('./IMG_3602_s.JPG')
# image = cv.imread('./IMG_3600_s.JPG')
image = cv.imread('./IMG_3674_s.jpg')

# Mask out everything outside the table with a hsv color scheme
hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
lower = np.array([110, 50, 50]) # Color code for dark blue
upper = np.array([130, 255, 255])
mask = cv.inRange(hsv_img, lower, upper)
table = cv.bitwise_and(image, image, mask = mask)
# cv.imshow("table", table)

# Convert to grayscale and find contours
table_gray = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
contours, hierarchy = cv.findContours(table_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# Find the largest contour as the border of the table
c = max(contours, key = cv.contourArea)
hull = cv.convexHull(c)
new_mask = np.zeros_like(image)
img_new = cv.drawContours(new_mask, [hull], -1, (255, 255, 255), -1)
cropped = cv.bitwise_and(image, img_new)
# cv.imshow("cropped", cropped)

# Mask the table out
balls = cv.bitwise_and(cropped, cropped, mask = 255-mask)
balls = cv.cvtColor(balls, cv.COLOR_BGR2GRAY)
# cv.imshow("balls", balls)
balls = cv.medianBlur(balls, 5)
# cv.imshow("balls", balls)

# Find balls
# Param1: higher = less circles
# Param2: higher = less circles
circles = cv.HoughCircles(balls, cv.HOUGH_GRADIENT, 1, 20, param1=11, param2=11, minRadius=10, maxRadius=14)
circles = np.uint16(np.around(circles))
print(circles)
print(circles.shape)
for i in circles[0, :]:
    cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# cv.imshow("images", np.hstack([cropped, image]))
cv.imshow("images", image)

cv.waitKey(0)