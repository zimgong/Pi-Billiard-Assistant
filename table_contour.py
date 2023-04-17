import numpy as np
import cv2 as cv

img = cv.imread('IMG_3602_s.JPG')
assert img is not None, "file could not be read, check with os.path.exists()"

# print(img.shape)
# print(img[int(img.shape[0]/2), int(img.shape[1]/2), :])
img = img - img[int(img.shape[0]/2), int(img.shape[1]/2), :]

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=40)
# circles = np.uint16(np.around(circles))
# print(circles)
# print(circles.shape)

# for i in circles[0, :]:
#     cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)


# ret, thresh = cv.threshold(img_gray, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# for i in contours:
#     (x, y), radius = cv.minEnclosingCircle(i)
#     center = (int(x), int(y))
#     radius = int(radius)
#     cv.circle(img, center, radius, (0, 255, 0), 2)

cv.imshow("Display window", img)
cv.waitKey(0)
cv.destroyAllWindows()