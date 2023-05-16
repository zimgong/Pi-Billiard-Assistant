#
# W_yz2874_zg284 5/9/2023 Raspberry Pi Main Script
# Description:
# This script is the main script for the raspberry pi. It will grab frames from the camera,
# detect the pool table, and send the table to the queue. It will also receive the lines
# from the queue and render them on the frame.
# The script uses multiprocessing to accelerate the process.
# The script also uses zmq to communicate with the laptop to send frames.
#

import argparse
import base64
import copy
import cv2 as cv
from datetime import datetime
import json
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
def simulate_stick(object1, num_iter, hull, lines, flag):
	while flag > 0:
		iter = 0
		ini_pos = copy.deepcopy(object1.pos)
		while iter <= num_iter:
			res = object1.move(hull)
			if res == False:
				flag -= 1
				lines.append([ini_pos[0], ini_pos[1], object1.pos[0], object1.pos[1]])
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

# Master process, grab frames from camera and show frames with rendered lines
def grab_frame_display(run_flag, frame_queue, line_queue, table_queue, dev, color):
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
		if initial:
			table = find_table(frame_hsv, color)
			if table is not None:
				hull = ConvexHull(table[:,0,:]) # Convert to ConvexHull object
				table_queue.put(hull) # Send table to queue
				table_queue.put(hull) # Send table to queue
				table_queue.put(hull) # Send table to queue
				print('P0 Put table, queue size: ', table_queue.qsize())
				initial = False
			else:
				print('P0 No table found')
				
		# Check if time since last send to queue exceeds 30ms
		curr_datetime = datetime.now()
		delta_time = curr_datetime-start_datetime
		delta_time_ms = delta_time.total_seconds()*1000
		if delta_time_ms > 10 and frame_queue.qsize() < 2 and table is not None: # If past time and queue is not full, send to queue
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
		if dev and (table is not None):
			cv.drawContours(frame, [table], -1, (255, 255, 255), 2) # Draw table contour
		if time.time() - last_receive_time < 1:
			for i in lines: # Draw lines to OpenCV frame
				print('P0 Trajectories, ', lines)
				cv.line(frame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 255), 2, cv.LINE_AA)
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
def process_stick_1(run_flag, frame_queue, table_queue, line_queue, dev, sensitivity):
	table = None
	while run_flag.value:
		if (table is None) and (not table_queue.empty()):
			table = table_queue.get()
		if not frame_queue.empty():
			frame_hsv = frame_queue.get() # Get frame from queue
			print('P1 Get frame, queue size: ', frame_queue.qsize())
			# color range for pick stick
			lower = np.array([0, 0, sensitivity])
			upper = np.array([255, 255-sensitivity, 255])

			mask = cv.inRange(frame_hsv, lower, upper)
			# Detect lines for stick
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
						cond2 = l[1] > 410 and l[3] > 410
						# cond3 = l[0] > 0 and l[0] < 160 and l[2] > 0 and l[2] < 170
						if not (cond1 or cond2):
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
					print('P1 Cue coordinates: ', cue) # Print to verify queue contents
				
					cue = np.array(cue, dtype=np.half)
					stick_euclid = np.linalg.norm(cue[2:4]-cue[0:2])/15
					vec = np.array((cue[2:4]-cue[0:2])/stick_euclid, dtype=np.half)
					obj_stick = Object(cue[2:4], 3, vec, 5)
					lines = []
					lines = simulate_stick(obj_stick, 100, table, lines, 2)
					print('P1 Trajectories:', lines)
					line_queue.put(lines)
					print('P1 Put line, queue size: ', line_queue.qsize())
			
		else:
			time.sleep(0.03)
	print("Quiting P1")
	print('P1 frame queue empty: ', frame_queue.empty())
	print('P1 stick queue empty: ', line_queue.empty())

# Process 2 detects the stick
def process_stick_2(run_flag, frame_queue, table_queue, line_queue, dev, sensitivity):
	table = None
	while run_flag.value:
		if (table is None) and (not table_queue.empty()):
			table = table_queue.get()
		if not frame_queue.empty():
			frame_hsv = frame_queue.get() # Get frame from queue
			print('P2 Get frame, queue size: ', frame_queue.qsize())
			# color range for pick stick
			lower = np.array([0, 0, sensitivity])
			upper = np.array([255, 255-sensitivity, 255])

			mask = cv.inRange(frame_hsv, lower, upper)
			# Detect lines for stick
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
						cond2 = l[1] > 410 and l[3] > 410
						# cond3 = l[0] > 0 and l[0] < 160 and l[2] > 0 and l[2] < 170
						if not (cond1 or cond2):
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
					print('P2 Cue coordinates: ', cue) # Print to verify queue contents
				
					cue = np.array(cue, dtype=np.half)
					stick_euclid = np.linalg.norm(cue[2:4]-cue[0:2])/15
					vec = np.array((cue[2:4]-cue[0:2])/stick_euclid, dtype=np.half)
					obj_stick = Object(cue[2:4], 3, vec, 5)
					lines = []
					lines = simulate_stick(obj_stick, 100, table, lines, 2)
					print('P2 Trajectories: ', lines)
					line_queue.put(lines)
					print('P2 Put line, queue size: ', line_queue.qsize())
			
		else:
			time.sleep(0.03)
	print("Quiting P2")
	print('P2 frame queue empty: ', frame_queue.empty())
	print('P2 stick queue empty: ', line_queue.empty())

# Process 3 computes Physics
def process_physics(run_flag, frame_queue, table_queue, line_queue, dev, sensitivity):
	table = None
	while run_flag.value:
		if (table is None) and (not table_queue.empty()):
			table = table_queue.get()
		if not frame_queue.empty():
			frame_hsv = frame_queue.get() # Get frame from queue
			print('P3 Get frame, queue size: ', frame_queue.qsize())
			# color range for pick stick
			lower = np.array([0, 0, sensitivity])
			upper = np.array([255, 255-sensitivity, 255])

			mask = cv.inRange(frame_hsv, lower, upper)
			# Detect lines for stick
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
						cond2 = l[1] > 410 and l[3] > 410
						# cond3 = l[0] > 0 and l[0] < 160 and l[2] > 0 and l[2] < 170
						if not (cond1 or cond2):
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
					print('P3 Cue coordinates: ', cue) # Print to verify queue contents
				
					cue = np.array(cue, dtype=np.half)
					stick_euclid = np.linalg.norm(cue[2:4]-cue[0:2])/15
					vec = np.array((cue[2:4]-cue[0:2])/stick_euclid, dtype=np.half)
					obj_stick = Object(cue[2:4], 3, vec, 5)
					lines = []
					lines = simulate_stick(obj_stick, 100, table, lines, 2)
					print('P3 Trajectories: ', lines)
					line_queue.put(lines)
					print('P3 Put line, queue size: ', line_queue.qsize())
			
		else:
			time.sleep(0.03)
	print("Quiting P3")
	print('P3 frame queue empty: ', frame_queue.empty())
	print('P3 stick queue empty: ', line_queue.empty())

RES_X = 640
RES_Y = 480
CENTER_X = RES_X/2
CENTER_Y = RES_Y/2

#Global Run Flag
frame = 0

if __name__ == '__main__':
    if args.mode == 'dev':
        dev = True
    else:
        dev = False
    with open('/home/pi/Project/cali.json') as json_file:
        cali = json.load(json_file)
    
    color = cali['color']
    sensitivity = cali['sensitivity']
    
    run_flag = Value('i', 1) 
    # run_flag controls all processes
    # initialize queues for inter-process communication
    frame_queue = Queue()
    stick_queue = Queue()
    line_queue = Queue()
    table_queue = Queue()
    # initialize processes
    p0 = Process(target=grab_frame_display, args=(run_flag, frame_queue, line_queue, table_queue, dev, color))
    p1 = Process(target=process_stick_1, args=(run_flag, frame_queue, table_queue, line_queue, dev, sensitivity))
    p2 = Process(target=process_stick_2, args=(run_flag, frame_queue, table_queue, line_queue, dev, sensitivity))
    p3 = Process(target=process_physics, args=(run_flag, frame_queue, table_queue, line_queue, dev, sensitivity))
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