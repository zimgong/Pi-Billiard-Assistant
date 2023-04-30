#
# W_yz2874_zg284 4/28/2023  
#

import cv2 as cv
import numpy as np
import copy

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
        if point_in_hull(self.pos, hull) == False:
            self.direct[0] = -self.direct[0]
            self.direct[1] = -self.direct[1]
            return False
        return True

# Check if a point is inside the convex hull
def point_in_hull(point, hull, tolerance=1e-12):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in hull.equations)

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
    while iter <= num_iter:
        res = object1.move(hull)
        if res == False:
            break
        collided = collide(object1, object2)
        if collided:
            change_v(object1, object2)
            cv.line(image, (ini_pos[0], ini_pos[1]), (object1.pos[0], object1.pos[1]), (255,255,255), 2, cv.LINE_AA)
            simulate_ball(object2, num_iter, image, hull)
            return
    cv.line(image, ini_pos, object1.pos, (255,255,255), 2, cv.LINE_AA)
    # print(ini_pos)
    # print(object.pos)
    return

# Simulate the cue ball behavior, move the cue ball
def simulate_ball(object, num_iter, image, hull):
    iter = 0
    ini_pos = copy.deepcopy(object.pos)
    while iter <= num_iter:
        res = object.move(hull)
        if res == False:
            break
    cv.line(image, (ini_pos[0], ini_pos[1]), (object.pos[0], object.pos[1]), (255,255,255), 2, cv.LINE_AA)
