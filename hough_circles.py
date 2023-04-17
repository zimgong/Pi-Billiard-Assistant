import numpy as np
import cv2 as cv

img = cv.imread('balls.jpg')
# img_bg = cv.imread('background.jpg')
# img = img - img_bg
assert img is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# imgray = cv.equalizeHist(imgray)
# imgray = cv.medianBlur(imgray, 5)
circles = cv.HoughCircles(imgray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=40, maxRadius=60)
circles = np.uint16(np.around(circles))
print(circles)
print(circles.shape)

for i in circles[0, :]:
    cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imshow("Display window", img)
cv.waitKey(0)
cv.destroyAllWindows()