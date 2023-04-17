import numpy as np
import cv2 as cv

img = cv.imread('coins.jpg')
img_bg = cv.imread('background.jpg')
img = img - img_bg
assert img is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgray = cv.equalizeHist(imgray)
imgray = cv.medianBlur(imgray, 5)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for i in contours:
    (x, y), radius = cv.minEnclosingCircle(i)
    center = (int(x), int(y))
    radius = int(radius)
    cv.circle(img, center, radius, (0, 255, 0), 2)

cv.imshow("Display window", img)
k = cv.waitKey(0)