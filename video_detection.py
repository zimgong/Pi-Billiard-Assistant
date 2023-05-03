import numpy as np
import argparse
import cv2 as cv

# cap = cv.VideoCapture('./3076_720p.mp4')
# lower = 80, 80, 0
# upper = 240, 150, 120

# cap = cv.VideoCapture('./696_720p.mp4')
# cap = cv.VideoCapture('./716_720p.mp4')
cap = cv.VideoCapture('./769_480p.mp4')

# Retake table map every 120 frames to avoid effect of hand movement
count = 0

def find_table(frame_hsv):
	# hsv color range for blue pool table
	lower_blue = np.array([110,50,50])
	upper_blue = np.array([130,255,255])
	# Mask out everything but the pool table (blue)
	mask = cv.inRange(frame_hsv, lower_blue, upper_blue)
	# Find the pool table contour
	contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
	# Find the largest contour as the border of the table
	try:
		table = max(contours, key = cv.contourArea)
	except:
		print("No table found!")
		return None
	return cv.convexHull(table)

while cap.isOpened():
    ret, frame = cap.read()
    
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    if count == 0:
        table = find_table(frame_hsv)

    new_mask = np.zeros_like(frame)
    img_new = cv.drawContours(new_mask, [table], -1, (255, 255, 255), -1)
    cropped = cv.bitwise_and(frame, img_new)

    lower = np.array([150, 120, 120])
    upper = np.array([165, 255, 255])
    mask = cv.inRange(frame_hsv, lower, upper)

    lines = cv.HoughLinesP(mask, 1, np.pi/180, 40, None, 20, 0)
    print("Line coordinates:", lines)
    cue = 0
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cue += l
            cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv.LINE_AA)
        cue = cue / len(lines)
        cue = np.round(cue).astype(int)

    sensitivity = 80
    hsv_img = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 255 - sensitivity])
    upper = np.array([255, sensitivity, 255])
    mask = cv.inRange(hsv_img, lower, upper)
    new_img = cv.bitwise_and(cropped, cropped, mask = mask)

    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, 20, param1=11, param2=11, minRadius=10, maxRadius=15)
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
            cv.circle(frame, (min[0], min[1]), min[2], (0, 255, 0), 2)
            cv.circle(frame, (min[0], min[1]), 2, (0, 0, 255), 3)
            cv.line(frame, (cue[2], cue[3]), (min[0], min[1]), (255,255,0), 2, cv.LINE_AA)

    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break

    count += 1
    if count >= 30:
        count = 0

cap.release()
cv.destroyAllWindows()