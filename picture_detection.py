import cv2 as cv
import numpy as np
import copy
from scipy.spatial import ConvexHull
import time

# RES_X = 640
# RES_Y = 480
# cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, RES_X)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, RES_Y)

# time.sleep(1)

# ret, frame = cap.read()

frame = cv.imread('frame_7.jpg')

# Define a class for balls and the move function
class Object:
    def __init__(self, pos, speed, direct, radius):
        self.pos = pos # Ball position
        self.speed = speed # Ball speed, not used yet
        self.direct = direct # Ball direction vector
        self.radius = radius # Ball radius

    def move(self, hull): # Move the ball based on its direction vector
        self.pos[0] += self.direct[0]
        self.pos[1] += self.direct[1]
        # If the ball is out of the table, change its direction, not yet implemented
        res = point_in_hull(self.pos, hull)
        if res is not None:
            self.direct = collide_hull(self.direct[0:2], res)
            return False
        return True

# Check if a point is inside the convex hull
def point_in_hull(point, hull, tolerance=1e-12):
        res = 0
        for eq in hull.equations:
            res = np.dot(eq[:-1], point) + eq[-1]
            if res >= tolerance:
                return eq[0:2]
        return None

def collide_hull(direct, eq):
    res = direct-2*np.dot(direct, eq)*eq 
    return  res  

# A helper function for collide detection between two objects
def collide(object1, object2):
    dist = np.linalg.norm(object1.pos-object2.pos)
    if dist <= (object1.radius + object2.radius) + 1:
        return True
    return False

# Change the speed of two objects based on real-world physics
def change_v(object1, object2):
    m1, m2 = object1.radius**2, object2.radius**2
    M = m1 + m2
    r1, r2 = object1.pos, object2.pos
    d = np.linalg.norm(r1-r2)**2
    v1 = object1.direct
    v2 = object2.direct
    # u1 = (v1 - 2*m2/M*np.dot(v1-v2,r1-r2)/d*(r1-r2))
    u2 = (v2 - 2*m1/M*np.dot(v2-v1,r2-r1)/d*(r2-r1))
    # object1.speed = [round(u1[0]), round(u1[1])]
    # We only care about the speed of the second object
    object2.direct = [u2[0], u2[1]]

# Simulate the cue stick behavior, hits the cue ball and let the cue ball move
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

def find_table(frame_hsv):
    # hsv color range for blue pool table
    lower_blue = np.array([100,150,155])
    upper_blue = np.array([120,255,255])
    # Mask out everything but the pool table (blue)
    # cv.imshow("frame_hsv", frame_hsv)
    mask = cv.inRange(frame_hsv, lower_blue, upper_blue)
    # cv.imshow("cropped table", mask)
    # Find the pool table contour
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # Find the largest contour as the border of the table
    try:
        table = max(contours, key = cv.contourArea)
    except:
        print("No table found!")
        return None
    return cv.convexHull(table)

frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

table = find_table(frame_hsv)

new_mask = np.zeros_like(frame)
img_new = cv.drawContours(new_mask, [table], -1, (255, 255, 255), -1)
cropped = cv.bitwise_and(frame_hsv, img_new)
# cv.imshow("cropped", cropped)

sensitivity = 80
lower = np.array([145, 10, 10])
upper = np.array([255, 255, 255])
mask = cv.inRange(cropped, lower, upper)
cv.imshow("mask", mask)
lines = cv.HoughLinesP(mask, 1, np.pi/180, 50, None, 20, 0)
# print("Line coordinates:", lines)
cue = 0
if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i][0]
        cue += l
        cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv.LINE_AA)
    cue = cue / len(lines)
    cue = cue.astype(int)
print(lines)

if lines is not None:
    center = np.array([320, 240])
    d0 = np.linalg.norm(cue[0:2] - center)
    d1 = np.linalg.norm(cue[2:4] - center)
    if d0 < d1:
        cue[0], cue[2] = cue[2], cue[0]
        cue[1], cue[3] = cue[3], cue[1]
    # print("Cue stick coordinates:", cue)

hull = ConvexHull(table[:,0,:]) # Turn the table coordinates into a convex hull

if cue is not 0:
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
    new_mask = np.zeros_like(frame)
    for i in lines:
        cv.line(frame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0,255,0), 2, cv.LINE_AA)

# cv.imshow("frame", frame)
# if cv.waitKey(1) == ord('q'):
#         cv.destroyAllWindows()

# cv.imwrite('frame.jpg', frame)
cv.imshow("frame", frame)
cv.waitKey(0)
# cap.release()
cv.destroyAllWindows()