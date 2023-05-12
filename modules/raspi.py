#
# W_yz2874_zg284 5/9/2023 Frame for Raspberry Pi Camera Send
#

import cv2 as cv
import zmq
import base64

IP = '10.48.155.12'

camera = cv.VideoCapture(0)
camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv.CAP_PROP_FPS, 30)
# rawCapture = PiRGBArray(camera, size=(640, 480))

context = zmq.Context()
footage_socket = context.socket(zmq.PAIR)

footage_socket.connect('tcp://' + IP + ':5555')
print(IP)

code_run = True

while code_run:
    ret, frame = camera.read()
    encoded, buffer = cv.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    footage_socket.send(jpg_as_text)