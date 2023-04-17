import numpy as np
import cv2 as cv
import sys

img = cv.imread('coins.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

background = cv.imread('background.jpg')

gray_bg = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

sub_img = gray_img - gray_bg

sub_img = cv.equalizeHist(sub_img)
sub_img = cv.medianBlur(sub_img,5)

ret, thresh = cv.threshold(sub_img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(sub_img, contours, -1, (0,255,0), 3)
cv.imshow("Display window", sub_img)
k = cv.waitKey(0)
