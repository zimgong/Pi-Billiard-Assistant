#
# W_yz2874_zg284 4/28/2023  
#

import cv2 as cv
import numpy as np
import copy
from scipy.spatial import ConvexHull

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

# # Check if a point is inside the convex hull
# def point_in_hull(point, hull, tolerance=1e-12):
#         return all(
#             (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
#             for eq in hull.equations)

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
def simulate_stick(object1, object2, num_iter, image, hull):
    iter = 0
    collided = False
    ini_pos = copy.deepcopy(object1.pos)
    lines = []
    while iter <= num_iter:
        res = object1.move(hull)
        if res == False:
            break
        collided = collide(object1, object2)
        if collided:
            change_v(object1, object2)
            lines.append([ini_pos[0], ini_pos[1], object1.pos[0], object1.pos[1]])
            # cv.line(image, (ini_pos[0], ini_pos[1]), (object1.pos[0], object1.pos[1]), (255,255,255), 2, cv.LINE_AA)
            lines = simulate_ball(object2, num_iter, image, hull, lines, True)
            return lines
    if collided == False:
        lines.append([ini_pos[0], ini_pos[1], object1.pos[0], object1.pos[1]])
        # cv.line(image, (ini_pos[0], ini_pos[1]), (object1.pos[0], object1.pos[1]), (255,255,255), 2, cv.LINE_AA)
    return lines

# Simulate the cue ball behavior, move the cue ball
def simulate_ball(object, num_iter, image, hull, lines, flag):
    iter = 0
    ini_pos = copy.deepcopy(object.pos)
    while iter <= num_iter:
        res = object.move(hull)
        if res == False:
            if flag == True:
                simulate_ball(object, num_iter, image, hull, lines, False)
            break
    lines.append([ini_pos[0], ini_pos[1], object.pos[0], object.pos[1]])
    # cv.line(image, (ini_pos[0], ini_pos[1]), (object.pos[0], object.pos[1]), (255,255,255), 2, cv.LINE_AA)
    return lines

# 

# print(point_in_hull(np.array([0, 0]), hull))

# # print(obj_stick.direct)
# # print(hull.equations[1][0:2])
# # print(collide_hull(obj_stick, hull, 1))

# # image = cv.imread('./IMG_3674_s.jpg')

# # cv.line(image, (cue_stick[0], cue_stick[1]), (cue_stick[2], cue_stick[3]), (0,0,0), 2, cv.LINE_AA)

# # cv.circle(image, (i[0], i[1]), 2, (255,255,255), 2, cv.LINE_AA)
# # cv.line(image, (cue_stick[0], cue_stick[1]), (cue_stick[2], cue_stick[3]), (0,0,0), 2, cv.LINE_AA)
# # cv.circle(image, (cue_ball[0], cue_ball[1]), cue_ball[2], (255,0,255), 2, cv.LINE_AA)

# # cv.imshow("image", image)
# # cv.waitKey(0)

# for i in range(len(hull.vertices)):
#     print(hull.vertices[i])
#     print(hull.equations[i])

# for i in hull.points:
#     print(i)