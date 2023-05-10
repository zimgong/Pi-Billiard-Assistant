import argparse
import base64
import copy
import cv2 as cv
from datetime import datetime
from multiprocessing import Process, Queue, Value
import numpy as np
from scipy.spatial import ConvexHull
import time
import zmq

parser = argparse.ArgumentParser(description='Main script for pool game detection on raspberry pi.')
parser.add_argument('--mode', type=str, help='user/developer mode', default='user')
args = parser.parse_args()

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
    lower_blue = np.array([100,120,120])
    upper_blue = np.array([120,255,255])
    # Mask out everything but the pool table (blue)
    mask = cv.inRange(frame_hsv, lower_blue, upper_blue)
    # cv.imshow("cropped table", mask)
    # Find the pool table contour
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # Find the largest contour as the border of the table
    try:
        table = max(contours, key = cv.contourArea)
    except:
        # print("No table found!")
        return None
    return cv.convexHull(table)

# Master process, grab frames from camera and show frames with rendered lines
def grab_frame_display(run_flag, frame_queue, line_queue, table_queue, dev):
	IP = '10.48.155.12'
	context = zmq.Context()
	footage_socket = context.socket(zmq.PAIR)
	footage_socket.connect('tcp://' + IP + ':5555')
	start_datetime = datetime.now() # Initialize start time
	last_receive_time = 0 # Initialize last receive time
	initial = True # Flag for first frame
	cap = cv.VideoCapture(0) # Test mode, load video
	cap.set(cv.CAP_PROP_FRAME_WIDTH, RES_X)
	cap.set(cv.CAP_PROP_FRAME_HEIGHT, RES_Y)
	cap.set(cv.CAP_PROP_FPS, 30) # Set frame rate, testing
	if not cap.isOpened():
		print("Cannot open camera")
		exit()
	while run_flag.value:
		ret, frame = cap.read() # Capture frame-by-frame
		# if frame is read correctly, ret is True
		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			cap.release()
			cv.destroyAllWindows()
			run_flag.value = 0
			break
		# Convert to hsv color space
		frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
		# Find the pool table if it is the first frame
		if initial or table is None:
			try:
				table = find_table(frame_hsv)
				hull = ConvexHull(table[:,0,:]) # Convert to ConvexHull object
				table_queue.put(hull) # Send table to queue
				print('P0 Put table, queue size: ', table_queue.qsize())
			except:
				print("No table found!")
			initial = False
		# Check if time since last send to queue exceeds 30ms
		curr_datetime = datetime.now()
		delta_time = curr_datetime-start_datetime
		delta_time_ms = delta_time.total_seconds()*1000
		if delta_time_ms > 10 and frame_queue.qsize() < 4: # If past time and queue is not full, send to queue
			start_datetime = curr_datetime # Update start time
			new_mask = np.zeros_like(frame)
			img_new = cv.drawContours(new_mask, [table], -1, (255, 255, 255), -1)
			cropped = cv.bitwise_and(frame_hsv, img_new)
			frame_queue.put(cropped) # Send frame to queue
			print('P0 Put frame, queue size: ', frame_queue.qsize())
		if not line_queue.empty(): # Receive lines from queue
			last_receive_time = time.time()
			lines = line_queue.get()
			print('P0 Get line, queue size: ', line_queue.qsize())
		if time.time() - last_receive_time < 0.5:
			for i in lines: # Draw lines to OpenCV frame
				cv.line(frame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 255), 2, cv.LINE_AA)
			if dev:
				cv.drawContours(frame, [table], -1, (255, 255, 255), 2) # Draw table contour
		_, buffer = cv.imencode('.jpg', frame)
		jpg_as_text = base64.b64encode(buffer)
		footage_socket.send(jpg_as_text)
		# cv.imshow('frame', frame) # Display the resulting frame
		if cv.waitKey(1) == ord('q'): # Press q to quit
			cap.release() # Release camera
			cv.destroyAllWindows() # Close all windows
			run_flag.value = 0
			print("Set run_flag 0, start quiting sequence")
			break
	frame_queue.put(None) # Send None to queue to signal other processes to quit
	print("Quiting P0")
	print('P0 frame queue empty: ', frame_queue.empty())
	print('P0 line queue empty: ', line_queue.empty())

# Process 1 detects the stick, it is exactly the same as Process 2
def process_stick_1(run_flag, frame_queue, stick_queue, dev):
	while run_flag.value:
		if not frame_queue.empty():
			frame_hsv = frame_queue.get() # Get frame from queue
			print('P1 Get frame, queue size: ', frame_queue.qsize())
			# color range for pick stick
			sensitivity = 150
			lower = np.array([0, 0, sensitivity])
			upper = np.array([255, 255-sensitivity, 255])
			try:
				mask = cv.inRange(frame_hsv, lower, upper)
				# Detect lines for stick
				lines = None
				start_length = 320
				while lines is None:
					start_length = start_length / 2
					lines = cv.HoughLinesP(mask, 1, np.pi/180, 120, None, minLineLength=start_length, maxLineGap=20)
					if start_length < 50:
						break
				# print("Line coordinates:", lines)
				cue = 0 # Take the average coordinates for the lines
				n = 0
				if lines is not None:
					for i in range(0, len(lines)):
						l = lines[i][0]
						cond1 = l[1] > 130 and l[1] < 160 and l[3] > 130 and l[3] < 160
						cond2 = l[1] > 340 and l[1] < 370 and l[3] > 340 and l[3] < 370
						cond3 = l[0] > 0 and l[0] < 60 and l[2] > 0 and l[2] < 60
						if not (cond1 or cond2 or cond3):
							cue += l
							n += 1
						cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv.LINE_AA)
					cue = cue / n
					cue = cue.astype(int) # Stick coordinates have to be integers to be printed on the frame
					# Decide the line direction by measuring the distance to the center of the frame
					# Take the endpoint closer to the center as the tip of the stick
					center = np.array([320, 240])
					d0 = np.linalg.norm(cue[0:2] - center)
					d1 = np.linalg.norm(cue[2:4] - center)
					if d0 < d1:
						cue[0], cue[2] = cue[2], cue[0]
						cue[1], cue[3] = cue[3], cue[1]
				stick_queue.put(cue) # Put stick coordinates to queue
				print('P1 Cue coordinates: ', cue) # Print to verify queue contents
				print('P1 Put stick, queue size: ', stick_queue.qsize())
			except:
				print("No stick found!")
		else:
			time.sleep(0.03)
	while not frame_queue.empty():
		x = frame_queue.get()
		if x is None:
			stick_queue.put(None)
	print("Quiting P1")
	print('P1 frame queue empty: ', frame_queue.empty())
	print('P1 stick queue empty: ', stick_queue.empty())

# Process 2 detects the stick
def process_stick_2(run_flag, frame_queue, stick_queue, dev):
	while run_flag.value:
		if not frame_queue.empty():
			frame_hsv = frame_queue.get() # Get frame from queue
			print('P2 Get frame, queue size: ', frame_queue.qsize())
			# color range for pick stick
			sensitivity = 150
			lower = np.array([0, 0, sensitivity])
			upper = np.array([255, 255-sensitivity, 255])
			try:
				mask = cv.inRange(frame_hsv, lower, upper)
				# Detect lines for stick
				lines = None
				start_length = 320
				while lines is None:
					start_length = start_length / 2
					lines = cv.HoughLinesP(mask, 1, np.pi/180, 120, None, minLineLength=start_length, maxLineGap=20)
					if start_length < 50:
						break
				print("Line coordinates:", lines)
				cue = 0 # Take the average coordinates for the lines
				n = 0
				if lines is not None:
					for i in range(0, len(lines)):
						l = lines[i][0]
						cond1 = l[1] > 130 and l[1] < 160 and l[3] > 130 and l[3] < 160
						cond2 = l[1] > 340 and l[1] < 370 and l[3] > 340 and l[3] < 370
						cond3 = l[0] > 0 and l[0] < 60 and l[2] > 0 and l[2] < 60
						if not (cond1 or cond2 or cond3):
							cue += l
							n += 1
						cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv.LINE_AA)
					cue = cue / n
					cue = cue.astype(int) # Stick coordinates have to be integers to be printed on the frame
					# Decide the line direction by measuring the distance to the center of the frame
					# Take the endpoint closer to the center as the tip of the stick
					center = np.array([320, 240])
					d0 = np.linalg.norm(cue[0:2] - center)
					d1 = np.linalg.norm(cue[2:4] - center)
					if d0 < d1:
						cue[0], cue[2] = cue[2], cue[0]
						cue[1], cue[3] = cue[3], cue[1]
				stick_queue.put(cue) # Put stick coordinates to queue
				print('P2 Cue coordinates: ', cue) # Print to verify queue contents
				print('P2 Put stick, queue size: ', stick_queue.qsize())
			except:
				print("No stick found!")
		else:
			time.sleep(0.03)
	while not frame_queue.empty():
		x = frame_queue.get()
		if x is None:
			stick_queue.put(None)
	print("Quiting P2")
	print('P2 frame queue empty: ', frame_queue.empty())
	print('P2 stick queue empty: ', stick_queue.empty())

# Process 3 computes Physics
def process_physics(run_flag, stick_queue, line_queue, table_queue, dev):
	table = []
	while run_flag.value:
		if not stick_queue.empty():
			if not table_queue.empty():
				table = table_queue.get()
			cue = stick_queue.get()
			if cue is None:
				break
			print('P3 Get stick, queue size: ', stick_queue.qsize())
			cue = np.array(cue, dtype=np.half)
			print('P3 Cue coordinates: ', cue)
			try:
				stick_euclid = np.linalg.norm(cue[2:4]-cue[0:2])/15
				vec = np.array((cue[2:4]-cue[0:2])/stick_euclid, dtype=np.half)
				obj_stick = Object(cue[2:4], 3, vec, 5)
				lines = []
				lines = simulate_stick(obj_stick, 100, frame, table, lines, 3)
				line_queue.put(lines)
				print('P3 Put line, queue size: ', line_queue.qsize())
			except:
				print("No stick detected")
		else:
			print("Processor 3 Didn't Receive Frame, sleep for 30ms")
			time.sleep(0.03)
	while not stick_queue.empty():
		x = stick_queue.get()
		if x is None:
			print("Quiting P3")
			print('P3 line queue empty: ', line_queue.empty())

RES_X = 640
RES_Y = 480
CENTER_X = RES_X/2
CENTER_Y = RES_Y/2
# cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, RES_X)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, RES_Y)

#Global Run Flag
table = None
frame = 0

if args.mode == 'dev':
	dev = True
else:
	dev = False

if __name__ == '__main__':
    run_flag = Value('i', 1) 
    # run_flag controls all processes
    # initialize queues for inter-process communication
    frame_queue = Queue()
    stick_queue = Queue()
    line_queue = Queue()
    table_queue = Queue()
    # initialize processes
    p0 = Process(target=grab_frame_display, args=(run_flag, frame_queue, line_queue, table_queue, dev))
    p1 = Process(target=process_stick_1, args=(run_flag, frame_queue, stick_queue, dev))
    p2 = Process(target=process_stick_2, args=(run_flag, frame_queue, stick_queue, dev))
    p3 = Process(target=process_physics, args=(run_flag, stick_queue, line_queue, table_queue, dev))
    # start processes
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    # wait for processes to finish
    p0.join()
    p1.join()
    p2.join()
    p3.join()