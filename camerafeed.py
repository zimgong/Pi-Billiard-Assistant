#
# W_yz2874_zg284 5/9/2023 Frame for Raspberry Pi Camera
#

import cv2 as cv
import zmq
import base64
import picamera
from picamera.array import PiRGBArray

IP = '10.48.155.12'

camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

context = zmq.Context()
footage_socket = context.socket(zmq.PAIR)

footage_socket.connect('tcp://' + IP + ':5555')
print(IP)

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    frame_image = frame.array
    encoded, buffer = cv.imencode('.jpg', frame_image)

    jpg_as_text = base64.b64encode(buffer)
    footage_socket.send(jpg_as_text)
    rawCapture.truncate(0)