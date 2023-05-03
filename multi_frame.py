from datetime import datetime
from multiprocessing import Process, Queue, Value
from scipy.spatial import ConvexHull
import copy
import cv2 as cv
import numpy as np
import time

# Time for initialization
time.sleep(1)

# Function for the Master Process
def grab_frame_display(run_flag, frame_queue, line_queue):
	start_datetime = datetime.now()
	while run_flag.value:
		# Capture frame-by-frame
		ret, frame = cap.read()
		# if frame is read correctly ret is True
		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			break
		# Check if time since last send to queue exceeds 30ms
		curr_datetime = datetime.now()
		delta_time = curr_datetime-start_datetime
		delta_time_ms = delta_time.total_seconds()*1000
		if delta_time_ms > 30 and frame_queue.qsize() < 4:
			start_datetime = curr_datetime
			frame_queue.put(frame)
			print('P0 Put one frame', frame_queue.qsize())
		if not line_queue.empty():
			last_line_receive_time = time.time()
			line = line_queue.get()
			print('P0 Get one line', line_queue.qsize())
		cv.imshow('frame',frame)
		if cv.waitKey(1) == ord('q'):
			cap.release()
			cv.destroyAllWindows()
			run_flag.value = 0
			print("set run_flag --- 0")
	while not line_queue.empty():
		line_queue.get()
		print('Get one line', line_queue.qsize())
	print("Quiting Main Processor")
	print('Main: fsq ', frame_queue.empty())

# Process 1 detects the stick
def process_stick(run_flag, frame_queue, stick_queue, p_start_turn):
	while run_flag.value:
		if not frame_queue.empty() and p_start_turn.value == 1:
			p_start_turn.value = 2
			frame_queue.get()
			print('P1 Get one frame', frame_queue.qsize())
			stick_queue.put(frame) # Put stick back
			print('P1 Put one stick', stick_queue.qsize())
		else:
			time.sleep(0.03)
	# while not frame_queue.empty():
	# 	frame_queue.get()
	# while not stick_queue.empty():
	# 	stick_queue.get()
	print("Quiting Stick Processor")
	print('sp: fsq ', frame_queue.empty())
	print('sp: sq ', stick_queue.empty())

# Process 2 detects the ball
def process_ball(run_flag, frame_queue, stick_queue, ball_queue, p_start_turn):
	while run_flag.value:
		if not frame_queue.empty() and not stick_queue.empty() and p_start_turn.value == 2:
			p_start_turn.value = 3
			frame = frame_queue.get()
			print('P2 Get one frame', frame_queue.qsize())
			stick = stick_queue.get()
			print('P2 Get one stick', stick_queue.qsize())
			ball_queue.put(frame) # Put stick back
			print('P2 Put one ball', ball_queue.qsize())
		else:
			time.sleep(0.03)
	# while not frame_queue.empty():
	# 	frame_queue.get()
	# while not stick_queue.empty():
	# 	stick_queue.get()
	# while not ball_queue.empty():
	# 	ball_queue.get()
	print("Quiting Ball Processor")
	print('bp: bq', ball_queue.empty())
	print('bp: fbq', frame_queue.empty())
	print('bp: sq', stick_queue.empty())


# Process 3 computes Physics
def process_physics(run_flag, ball_queue, line_queue, p_start_turn):
	while run_flag.value:
		if not ball_queue.empty() and p_start_turn.value == 3:
			p_start_turn.value = 1
			frame = ball_queue.get()
			print('P3 Get one ball', ball_queue.qsize())
			line_queue.put(frame) # Put stick back
			print('P3 Put one line', line_queue.qsize())
		else:
			#print("Processor 3 Didn't Receive Frame, sleep for 30ms")
			time.sleep(0.03)
	# while not ball_queue.empty():
	# 	ball_queue.get()
	print("Quiting Physics Processor")
	print('physp: bq', ball_queue.empty())
	print('physp: lq', line_queue.empty())

RES_X = 640
RES_Y = 480
CENTER_X = RES_X/2
CENTER_Y = RES_Y/2
cap = cv.VideoCapture(0)
if not cap.isOpened():
	print("Cannot open camera")
	exit()
cap.set(cv.CAP_PROP_FRAME_WIDTH, RES_X)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, RES_Y)

#Global Run Flag
table = 0
frame = 0

if __name__ == '__main__':
	run_flag = Value('i', 1) 
	# p_start_turn is used to keep worker processes process in order
	p_start_turn = Value('i', 1)  
	frame_queue = Queue()
	stick_queue = Queue()
	ball_queue = Queue()
	line_queue = Queue()
	p0 = Process(target=grab_frame_display, args=(run_flag, frame_queue, line_queue))
	p1 = Process(target=process_stick, args=(run_flag, frame_queue, stick_queue, p_start_turn))
	p2 = Process(target=process_ball, args=(run_flag, frame_queue, stick_queue, ball_queue, p_start_turn))
	p3 = Process(target=process_physics, args=(run_flag, ball_queue, line_queue, p_start_turn))
	p0.start()
	p1.start()
	p2.start()
	p3.start()
	p0.join()
	p1.join()
	p2.join()
	p3.join()