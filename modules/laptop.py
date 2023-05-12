#
# W_yz2874_zg284 5/9/2023 Frame for Laptop Receive
#

import cv2 as cv
import zmq
import base64
import numpy as np

context = zmq.Context()

footage_socket = context.socket(zmq.PAIR)
footage_socket.bind('tcp://*:5555')

while True:
    print('Receiving...')
    frame = footage_socket.recv_string()
    img = base64.b64decode(frame)
    npimg = np.fromstring(img, dtype=np.uint8)
    source = cv.imdecode(npimg, 1)
    cv.imshow('Stream', source)
    if cv.waitKey(1) == ord('q'): # Press q to quit
        cv.destroyAllWindows() # Close all windows
        print("Connection closed")
        break