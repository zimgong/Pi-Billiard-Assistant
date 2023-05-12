#
# W_yz2874_zg284 5/9/2023 Pi Camera Picture Test
#

import copy
import cv2 as cv
import json
import numpy as np

from scipy.spatial import ConvexHull

RES_X = 640
RES_Y = 480
# cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, RES_X)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, RES_Y)

with open('cali.json') as json_file:
    cali = json.load(json_file)

color = cali['color']
sensitivity = cali['sensitivity']

# ret, frame = cap.read()

frame = cv.imread('frame_1.jpg')

# Define a class for balls and the move function
class Object:
    def __init__(self, pos, speed, direct, radius):
        self.pos = pos # Ball position
        self.speed = speed # Ball speed, not used yet
        self.direct = direct # Ball direction vector
        self.radius = radius # Ball radius

    def move(self, hull): # Move the ball based on its direction vector
        self.pos += self.direct
        # If the ball is out of the table, change its direction, not yet implemented
        res = point_in_hull(self.pos + 2 * self.direct, hull)
        if res is not None:
            self.direct = collide_hull(self.direct[0:2], res)
            return False
        return True

# Check if a point is inside the convex hull
# Return the normal vector of the plane that the point is outside the hull
# Return None if the point is inside
def point_in_hull(point, hull, tolerance=1e-12):
        res = 0
        for eq in hull.equations:
            res = np.dot(eq[:-1], point) + eq[-1]
            if res >= tolerance:
                return eq[0:2]
        return None

# Calculate the new direction vector after collision
def collide_hull(direct, eq):
    res = direct-2*np.dot(direct, eq)*eq 
    return  res  

# Simulate the cue stick behavior, hits the cue ball and let the cue ball move
# Return the lines that the cue stick has traveled
def simulate_stick(object1, num_iter, image, hull, lines, flag):
    iter = 0
    ini_pos = copy.deepcopy(object1.pos)
    while iter <= num_iter:
        res = object1.move(hull)
        if res == False:
            if flag > 0:
                flag -= 1
                lines.append([ini_pos[0], ini_pos[1], object1.pos[0], object1.pos[1]])
                lines = simulate_stick(object1, num_iter, image, hull, lines, flag)
            break
    return lines

# Find the pool table contour
# Return the cv.convexHull object of the table
# Return None if no table is found
def find_table(frame_hsv, color):
    # hsv color range for blue pool table
    lower_blue = np.array([color[0],120,120])
    upper_blue = np.array([color[1],255,255])
    # Mask out everything but the pool table (blue)
    mask = cv.inRange(frame_hsv, lower_blue, upper_blue)
    # cv.imshow("cropped table", mask)
    # Find the pool table contour
    contours = []
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # Find the largest contour as the border of the table
    if contours != []:
        table = max(contours, key = cv.contourArea)
        return cv.convexHull(table)
    else:
        return None

frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

table = find_table(frame_hsv, color)

new_mask = np.zeros_like(frame)
img_new = cv.drawContours(new_mask, [table], -1, (255, 255, 255), -1)
cropped = cv.bitwise_and(frame_hsv, img_new)
# cv.imshow("cropped", cropped)

lower = np.array([0, 0, sensitivity])
upper = np.array([255, 255-sensitivity, 255])
mask = cv.inRange(cropped, lower, upper)
# cv.imwrite('mask.jpg', mask)
# cv.imshow("mask", mask)
lines = None
start_length = 320

cue = np.array([0, 0, 0, 0])
flag = True
while not cue.any() or start_length > 15:
    cue = np.array([0, 0, 0, 0])
    start_length = start_length / 2

    if start_length < 40:
        lines = cv.HoughLinesP(mask, 1, np.pi/180, 40, None, minLineLength=15, maxLineGap=20)
        flag = False
    else:
        lines = cv.HoughLinesP(mask, 1, np.pi/180, 120, None, minLineLength=start_length, maxLineGap=20)

    if lines is not None:
        n = 0
        for i in range(0, len(lines)):
            l = lines[i][0]
            cond1 = l[1] < 160 and l[3] < 160
            cond2 = l[1] > 340 and l[1] < 370 and l[3] > 340 and l[3] < 370
            cond3 = l[0] > 0 and l[0] < 160 and l[2] > 0 and l[2] < 170
            if not (cond1 or cond2 or cond3):
                cue += l
                n += 1
                cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv.LINE_AA)
                break
    if flag == False:
        break
        
if cue.any():
    cue = cue / n
    cue = cue.astype(int)
    center = np.array([320, 240])
    d0 = np.linalg.norm(cue[0:2] - center)
    d1 = np.linalg.norm(cue[2:4] - center)
    if d0 < d1:
        cue[0], cue[2] = cue[2], cue[0]
        cue[1], cue[3] = cue[3], cue[1]
    # print("Cue stick coordinates:", cue)

    hull = ConvexHull(table[:,0,:]) # Turn the table coordinates into a convex hull

    # print(hull.points)
    print('Found cue!', cue)
    # print('Found balls!', min)
    cue = np.array(cue, dtype=np.half)
    stick_euclid = np.linalg.norm(cue[2:4]-cue[0:2])/15
    vec = np.array((cue[2:4]-cue[0:2])/stick_euclid, dtype=np.half)
    obj_stick = Object(cue[2:4], 3, vec, 5)
    lines = []
    lines = simulate_stick(obj_stick, 100, frame, hull, lines, 3)
    print(lines)
    for i in lines:
        cv.line(frame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0,255,0), 2, cv.LINE_AA)

# cv.drawContours(frame, [table], -1, (255, 255, 255), 2)

cv.imshow("frame", frame)
# if cv.waitKey(1) == ord('q'):
#     cv.destroyAllWindows()

cv.waitKey(0)

# cap.release()
cv.destroyAllWindows()