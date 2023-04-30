#
# W_yz2874_zg284 4/28/2023  
#

import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull
import cue_detection as cd
import ball_collision as bc

# Main function

# image = cv.imread('./IMG_3646_s.jpg')
# image = cv.imread('./IMG_3660_s.jpg')
# image = cv.imread('./IMG_3670_s.jpg')
image = cv.imread('./IMG_3674_s.jpg')
# image = cv.imread('./IMG_3681_s.jpg') # Failed, too less cue in picture
# image = cv.imread('./IMG_3684_s.jpg') # Failed, too less cue in picture
# image = cv.imread('./IMG_3686_s.jpg') # Failed, too much hand in picture and bad table contour
# image = cv.imread('./IMG_3692_s.jpg') # Failed, weird pole position

# A sample list of objects from detection file
# cue_ball = np.array([240, 660, 11])
# cue_stick = np.array([135, 670, 185, 661])
# table = np.array([[1215, 620],[1179, 680], [1163, 682],
#                   [80, 809], [59, 810], [58, 810],
#                   [12, 774], [12, 263], [56, 223], 
#                   [77, 217], [100, 214], [526, 161], 
#                   [1043, 98], [1047, 98], [1055, 101], 
#                   [1108, 135], [1114, 145], [1215, 608]
#                   ])

try:
    cue_stick, cue_ball, table = cd.detect_cue(image)

    # Draw out the table, cue stick and cue ball
    cv.drawContours(image, [table], -1, (255,255,255), 2)
    for i in table:
        cv.circle(image, (i[0], i[1]), 2, (255,255,255), 2, cv.LINE_AA)
    cv.line(image, cue_stick[0:2], cue_stick[2:4], (0,0,0), 2, cv.LINE_AA)
    cv.circle(image, (cue_ball[0], cue_ball[1]), cue_ball[2], (255,0,255), 2, cv.LINE_AA)

    hull = ConvexHull(table) # Turn the table coordinates into a convex hull
    stick_euclid = np.linalg.norm(cue_stick[2:4]-cue_stick[0:2])/10
    obj_stick = bc.Object(cue_stick[2:4], 3, (cue_stick[2:4]-cue_stick[0:2])/stick_euclid, 4)
    obj_ball = bc.Object(cue_ball[0:2], 0, [0,0], cue_ball[2])
    bc.simulate_stick(obj_stick, obj_ball, 100, image, hull)

    cv.imshow("image", image)
    cv.waitKey(0)

except:
    print("No cue stick or cue ball detected! ")