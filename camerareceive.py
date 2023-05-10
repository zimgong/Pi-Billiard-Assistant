#
# W_yz2874_zg284 5/9/2023 Frame for Laptop Camera
#

import cv2 as cv
import zmq
import base64
import numpy as np

context = zmq.Context()

footage_socket = context.socket(zmq.PAIR)
footage_socket.bind('tcp://*:5555')

while True:
    print('listion')
    frame = footage_socket.recv_string()
    img = base64.b64decode(frame)
    npimg = np.fromstring(img, dtype=np.uint8)
    source = cv.imdecode(npimg, 1)
    cv.imshow('Stream', source)
    cv.waitKey(1)