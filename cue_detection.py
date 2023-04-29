import numpy as np
import cv2 as cv

# image = cv.imread('./IMG_3602_s.JPG')
# image = cv.imread('./IMG_3671_s.jpg')
image = cv.imread('./IMG_3674_s.jpg')

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
cv.imshow("cropped table", table)

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
cv.imshow("hull", cropped)

hsv_img = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
lower = np.array([150, 120, 120])
upper = np.array([165, 255, 255])
mask = cv.inRange(hsv_img, lower, upper)
new_img = cv.bitwise_and(cropped, cropped, mask = mask)
# cv.imshow("new img", new_img)

table = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
lines = cv.HoughLinesP(table, 1, np.pi/180, 50, None, 50, 10)
# print(lines)

if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv.LINE_AA)
# cv.imshow("images", image)

sensitivity = 55
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
circles = cv.HoughCircles(gray_img, cv.HOUGH_GRADIENT, 1, 20, param1=11, param2=11, minRadius=9, maxRadius=12)
circles = np.uint16(np.around(circles))
# print(circles)
for i in circles[0, :]:
    cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
# cv.imshow("images", image)

cue = 0
for i in lines:
    cue += i
cue = cue / len(lines)

cv.line(image, (lines[1][0][0], lines[1][0][1]), (circles[0][0][0], circles[0][0][1]), (0,0,0), 2, cv.LINE_AA)


cv.waitKey(0)