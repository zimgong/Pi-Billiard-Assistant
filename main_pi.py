from datetime import datetime
from multiprocessing import Process, Queue, Value
from scipy.spatial import ConvexHull
import copy
import cv2 as cv
import numpy as np
import time
# import RPi.GPIO as GPIO
import pygame
from pygame.locals import * # for event MOUSE variables

def GPIO26_callback(channel): # GPIO 26 quit button
    global run_flag
    global code_run
    run_flag.value = not run_flag.value
    code_run = not code_run
    # GPIO.cleanup()
    print("Finished! Quit...")

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
        res = point_in_hull(self.pos + 3 * self.direct, hull)
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
    lower_blue = np.array([100,100,100])
    upper_blue = np.array([130,255,255])
    # Mask out everything but the pool table (blue)
    mask = cv.inRange(frame_hsv, lower_blue, upper_blue)
    # cv.imshow("cropped table", mask)
    # Find the pool table contour
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # Find the largest contour as the border of the table
    try:
        table = max(contours, key = cv.contourArea)
    except:
        print("No table found!")
        return None
    return cv.convexHull(table)

# Master process, grab frames from camera and show frames with rendered lines
def grab_frame_display(run_flag, frame_queue, line_queue, table_queue):
	start_datetime = datetime.now() # Initialize start time
	last_receive_time = 0 # Initialize last receive time
	initial = True # Flag for first frame
	cap = cv.VideoCapture('806_480P.mp4') # Test mode, load video
	# cap.set(cv.CAP_PROP_FPS, 20) # Set frame rate, testing
	cv.namedWindow('frame', cv.WINDOW_NORMAL) # Display full screen
	cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
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
			try:
				table = find_table(frame_hsv)
				table = ConvexHull(table[:,0,:]) # Convert to ConvexHull object
				table_queue.put(table) # Send table to queue
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
			frame_queue.put(frame_hsv) # Send frame to queue
			print('P0 Put frame, queue size: ', frame_queue.qsize())
		if not line_queue.empty(): # Receive lines from queue
			last_receive_time = time.time()
			lines = line_queue.get()
			print('P0 Get line, queue size: ', line_queue.qsize())
		if time.time() - last_receive_time < 0.5:
			blank = np.zeros_like(frame)
			blank[:] = (255, 0, 0)
			print('P0 Draw lines')
			for i in lines: # Draw lines to OpenCV frame
				cv.line(blank, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 255), 2, cv.LINE_AA)
			cv.imshow('frame', blank) # Display the resulting frame
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
def process_stick_1(run_flag, frame_queue, stick_queue):
	while run_flag.value:
		if not frame_queue.empty():
			frame_hsv = frame_queue.get() # Get frame from queue
			print('P1 Get frame, queue size: ', frame_queue.qsize())
			# color range for pick stick
			lower = np.array([140, 50, 50])
			upper = np.array([170, 255, 255])

			# lower = np.array([145, 5, 5])
			# upper = np.array([260, 255, 255])
			mask = cv.inRange(frame_hsv, lower, upper)
			# Detect lines for stick
			lines = cv.HoughLinesP(mask, 1, np.pi/180, 40, None, 20, 0)
			# print("Line coordinates:", lines)
			cue = 0 # Take the average coordinates for the lines
			if lines is not None:
				for i in range(0, len(lines)):
					l = lines[i][0]
					cue += l
					cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv.LINE_AA)
				cue = cue / len(lines)
				cue = cue.astype(int) # Stick coordinates have to be integers to be printed on the frame
			# print("Line coordinates:", lines)
			if lines is not None: 
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
def process_stick_2(run_flag, frame_queue, stick_queue):
	while run_flag.value:
		if not frame_queue.empty():
			# Get frame from queue
			frame_hsv = frame_queue.get()
			print('P1 Get frame, queue size: ', frame_queue.qsize())
			lower = np.array([140, 50, 50])
			upper = np.array([170, 255, 255])
			mask = cv.inRange(frame_hsv, lower, upper)
			# cv.imshow("mask", mask)

			lines = cv.HoughLinesP(mask, 1, np.pi/180, 40, None, 20, 0)
			# print("Line coordinates:", lines)
			cue = 0
			if lines is not None:
				for i in range(0, len(lines)):
					l = lines[i][0]
					cue += l
					cv.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv.LINE_AA)
				cue = cue / len(lines)
				cue = cue.astype(int)
			# print(lines)
			if lines is not None:
				center = np.array([320, 240])
				d0 = np.linalg.norm(cue[0:2] - center)
				d1 = np.linalg.norm(cue[2:4] - center)
				if d0 < d1:
					cue[0], cue[2] = cue[2], cue[0]
					cue[1], cue[3] = cue[3], cue[1]
			stick_queue.put(cue) # Put stick coordinates to queue
			print('P2 Cue coordinates: ', cue)
			print('P2 Put stick, queue size: ', stick_queue.qsize())
		else:
			time.sleep(0.03)
	while not frame_queue.empty():
		x = frame_queue.get()
		if x is None:
			stick_queue.put(None)
	print("Quiting P1")
	print('P2 frame queue empty: ', frame_queue.empty())
	print('P2 stick queue empty: ', stick_queue.empty())

# Process 3 computes Physics
def process_physics(run_flag, stick_queue, line_queue, table_queue):
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
			# print("Processor 3 Didn't Receive Frame, sleep for 30ms")
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
table = []
frame = 0

if __name__ == '__main__':
    code_run = True
    # GPIO.setmode(GPIO.BCM) # Initialize GPIO mode and setup input pin
    # GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    # GPIO.add_event_detect(26, GPIO.FALLING, callback=GPIO26_callback, bouncetime=300)
    # Initialize pygame and set general parameters
    pygame.init() # MUST occur AFTER os enviroment variable calls
    pygame.mouse.set_visible(True)
    WHITE = 255, 255, 255
    BLACK = 0, 0, 0
    screen = pygame.display.set_mode((640, 480)) # Set screen size
    
    # Define the buttons and the font
    my_font = pygame.font.Font('From Cartoon Blocks.ttf', 50) # Font size 50
    my_buttons = {'quit':(320, 400)} # Button dictionary
    screen.fill(BLACK) # Erase the work space
    
    header_font = pygame.font.Font('From Cartoon Blocks.ttf', 60)
    header_text = header_font.render("Pi Billiard Assistant", True, WHITE)
    header_rect = header_text.get_rect(center=(320, 120))
    screen.blit(header_text, header_rect)
    
    script_font = pygame.font.Font('From Cartoon Blocks.ttf', 40)
    script_text = script_font.render("Press the button to start", True, WHITE)
    script_rect = script_text.get_rect(center=(320, 320))
    screen.blit(script_text, script_rect)
    
    ball_img = pygame.transform.scale(pygame.image.load('balls.png'), (100, 100))
    ball_rect = ball_img.get_rect()
    ball_rect = ball_rect.move((390, 160))
    screen.blit(ball_img, ball_rect)
    stick_img = pygame.transform.scale(pygame.image.load('stick.png'), (200, 50))
    stick_rect = stick_img.get_rect()
    stick_rect = stick_rect.move((150, 220))
    screen.blit(stick_img, stick_rect)
    
    # Initialize the button and display it on the screen
    my_buttons_rect = {}
    for my_text, text_pos in my_buttons.items():
        text_surface = my_font.render(my_text, True, WHITE)
        rect = text_surface.get_rect(center=text_pos)
        screen.blit(text_surface, rect)
        my_buttons_rect[my_text] = rect # save rect for 'my-text' button
    
    pygame.display.flip()
    run_flag = Value('i', 0) 

    while code_run: # main loop
        for event in pygame.event.get(): # for detecting an event for touch screen...
            if (event.type == MOUSEBUTTONDOWN):
                pos = pygame.mouse.get_pos()
            elif (event.type == MOUSEBUTTONUP):
                pos = pygame.mouse.get_pos()
                for (my_text, rect) in my_buttons_rect.items(): # for saved button rects...
                    if (rect.collidepoint(pos)): # if collide with mouse click...
                        if (my_text == 'quit'): # indicate correct button press
                            code_run = False
                            run_flag.value = 1
    
    # run_flag controls all processes
    # initialize queues for inter-process communication
    frame_queue = Queue()
    stick_queue = Queue()
    line_queue = Queue()
    table_queue = Queue()
    # initialize processes
    p0 = Process(target=grab_frame_display, args=(run_flag, frame_queue, line_queue, table_queue))
    p1 = Process(target=process_stick_1, args=(run_flag, frame_queue, stick_queue))
    p2 = Process(target=process_stick_2, args=(run_flag, frame_queue, stick_queue))
    p3 = Process(target=process_physics, args=(run_flag, stick_queue, line_queue, table_queue))
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