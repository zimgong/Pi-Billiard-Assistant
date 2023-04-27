import numpy as np
import argparse
import cv2 as cv

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "Path to the image")
# args = vars(ap.parse_args())

# image = cv.imread(args["image"])

# image = cv.imread('./blue_table.png')
image = cv.imread('./IMG_3674_s.jpg')
# image = cv.imread('./IMG_3600_s.JPG')

# Hard code color range for table and mask out everything else
# Should work for blue or relevant colored table

# These are perfect for light blue! 
# lower = 100, 100, 0
# upper = 240, 160, 120

# These are for dark blue! 
lower = 50, 50, 0
upper = 240, 150, 120

# Mask out everything outside the table
mask = cv.inRange(image, lower, upper)
table = cv.bitwise_and(image, image, mask = mask)
cv.imshow("image", table)

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
lower = 90, 90, 190
upper = 255, 255, 255
mask = cv.inRange(image, lower, upper)
table = cv.bitwise_and(cropped, cropped, mask = mask)
# balls = cv.bitwise_and(cropped, cropped, mask = 255-mask)
# cv.imshow("images", table)
table = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
# balls = cv.medianBlur(balls, 5)
# cv.imshow("images", balls)

# # Find balls
# # Param1: higher = less circles
# # Param2: higher = less circles
contours, hierarchy = cv.findContours(table, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
c = max(contours, key = cv.contourArea)
img_new = cv.drawContours(image, c, -1, (255, 255, 255), 2)

# lines = cv.HoughLines(table, 1, np.pi/180, 360, None, 0, 0)

# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv.line(image, pt1, pt2, (0,0,0), 2, cv.LINE_AA)

# lines = cv.HoughLinesP(table, 1, np.pi/180, 100, None, 50, 10)

# if lines is not None:
#     for i in range(0, len(lines)):
#         l = lines[i][0]
#         cv.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv.LINE_AA)

# # # cv.imshow("images", np.hstack([cropped, image]))
cv.imshow("images", image)

cv.waitKey(0)