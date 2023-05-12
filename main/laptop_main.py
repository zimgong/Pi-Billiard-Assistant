#
# W_yz2874_zg284 5/10/2023 Laptop Main Script
#

import base64
import cv2 as cv
import json
import numpy as np
import os
import pygame
from pygame.locals import * # for event MOUSE variables
import zmq

# Initialize pygame and set general parameters
pygame.init() # MUST occur AFTER os enviroment variable calls
pygame.mouse.set_visible(True) # Set mouse visibility
RES_X = 640 # Set screen parametersns +
RES_Y = 480
WHITE = 255, 255, 255 # Set colors
BLACK = 0, 0, 0
screen = pygame.display.set_mode((RES_X, RES_Y)) # Set screen size

# Define the fonts
font_s = pygame.font.Font('From Cartoon Blocks.ttf', 40)
font_m = pygame.font.Font('From Cartoon Blocks.ttf', 50) # Font size 50
font_l = pygame.font.Font('From Cartoon Blocks.ttf', 60)

# Define the buttons
buttons_list = {'start':(RES_X/2, 325), 'calibrate':(RES_X/2, 375), 'quit':(RES_X/2, 425)} # Button dictionary
buttons_cali_t = {'blue':(450, 175), 'green':(450, 275), 'done':(RES_X/2, 425)}
buttons_cali_s = {'increase +':(450, 175), 'decrease -':(450, 275), 'done':(RES_X/2, 425)}

# Graphics
ball_img = pygame.transform.scale(pygame.image.load('balls.png'), (100, 100))
ball_rect = ball_img.get_rect()
ball_rect = ball_rect.move((390, 130))

stick_img = pygame.transform.scale(pygame.image.load('stick.png'), (200, 50))
stick_rect = stick_img.get_rect()
stick_rect = stick_rect.move((150, 190))

code_run = True
ui_run = True
calibrate = False
rpi_run = False
dev = False

while code_run:
    screen.fill(BLACK) # Erase the work space
    # Define the header and display it on the screen
    header_text = font_l.render("Pi Billiard Assistant", True, WHITE)
    header_rect = header_text.get_rect(center=(RES_X/2, 90))
    screen.blit(header_text, header_rect)
    screen.blit(ball_img, ball_rect)
    screen.blit(stick_img, stick_rect)
    # Initialize the button and display it on the screen
    button_rects = {}
    for text, text_pos in buttons_list.items():
        text_surface = font_m.render(text, True, WHITE)
        rect = text_surface.get_rect(center=text_pos)
        screen.blit(text_surface, rect)
        button_rects[text] = rect # save rect for 'my-text' button
    text_surface = font_s.render('mode: user', True, WHITE)
    dev_rect = text_surface.get_rect(center=(RES_X/2, 275))
    screen.blit(text_surface, dev_rect)
    pygame.display.flip()

    while ui_run:
        for event in pygame.event.get(): # for detecting an event for touch screen...
            if (event.type == MOUSEBUTTONUP):
                pos = pygame.mouse.get_pos()
                if dev_rect.collidepoint(pos):
                    dev = not dev
                    screen.fill(BLACK) # Erase the work space
                    screen.blit(header_text, header_rect)
                    screen.blit(ball_img, ball_rect)
                    screen.blit(stick_img, stick_rect)
                    for text, text_pos in buttons_list.items():
                        text_surface = font_m.render(text, True, WHITE)
                        rect = text_surface.get_rect(center=text_pos)
                        screen.blit(text_surface, rect)
                    if dev:
                        text_surface = font_s.render('mode: dev', True, WHITE)
                    else:
                        text_surface = font_s.render('mode: user', True, WHITE)
                    screen.blit(text_surface, dev_rect)
                    pygame.display.flip()
                for (text, rect) in button_rects.items(): # for saved button rects...
                    if (rect.collidepoint(pos)): # if collide with mouse click...
                        if (text == 'start'): # indicate correct button press
                            ui_run = False
                            rpi_run = True
                            pygame.quit()
                        elif (text == 'calibrate'): # indicate correct button press
                            ui_run = False
                            calibrate = True
                        elif (text == 'quit'): # indicate correct button press
                            ui_run = False
                            code_run = False

    if rpi_run:
        print('Starting Pi Billiard Assistant...')
        if dev:
            os.system('bash ./startpi_dev.sh')
        else:
            os.system('bash ./startpi.sh')
        context = zmq.Context()
        footage_socket = context.socket(zmq.PAIR)
        footage_socket.bind('tcp://*:5555')

        while True:
            # print('Receiving...')
            frame = footage_socket.recv_string()
            img = base64.b64decode(frame)
            npimg = np.fromstring(img, dtype=np.uint8)
            source = cv.imdecode(npimg, 1)
            cv.imshow('Stream', source)
            if cv.waitKey(1) == ord('q'): # Press q to quit
                cv.destroyAllWindows() # Close all windows
                print("Connection closed")
                ui_run = True
                os.system('bash ./killpi.sh')
                break
    
    if calibrate:
        color = [100, 120]
        sens = 175
        cali_stage = 0
        screen.fill(BLACK) # Erase the work space
        header_text = font_l.render("Calibrate Table", True, WHITE)
        header_rect = header_text.get_rect(center=(RES_X/2, 60))
        screen.blit(header_text, header_rect)
        cali_t_rects = {}
        for text, text_pos in buttons_cali_t.items():
            text_surface = font_s.render(text, True, WHITE)
            rect = text_surface.get_rect(center=text_pos)
            screen.blit(text_surface, rect)
            cali_t_rects[text] = rect # save rect for 'my-text' button
        text_surface = font_s.render('table color', True, WHITE)
        rect = text_surface.get_rect(center=(200, 150))
        screen.blit(text_surface, rect)
        text_surface = font_m.render('BLUE', True, WHITE)
        rect = text_surface.get_rect(center=(200, 225))
        screen.blit(text_surface, rect)
        pygame.display.flip()
        while cali_stage == 0:
            for event in pygame.event.get(): # for detecting an event for touch screen...
                if (event.type == MOUSEBUTTONUP):
                    pos = pygame.mouse.get_pos()
                    for (text, rect) in cali_t_rects.items(): # for saved button rects...
                        if (rect.collidepoint(pos)): # if collide with mouse click...
                            if (text == 'blue'): # indicate correct button press
                                color = [100, 120]
                                screen.fill(BLACK) # Erase the work space
                                for text, text_pos in buttons_cali_t.items():
                                    text_surface = font_s.render(text, True, WHITE)
                                    rect = text_surface.get_rect(center=text_pos)
                                    screen.blit(text_surface, rect)
                                screen.blit(header_text, header_rect)
                                text_surface = font_m.render('BLUE', True, WHITE)
                                rect = text_surface.get_rect(center=(200, 225))
                                screen.blit(text_surface, rect)
                                text_surface = font_s.render('table color', True, WHITE)
                                rect = text_surface.get_rect(center=(200, 150))
                                screen.blit(text_surface, rect)
                                pygame.display.flip()
                            elif (text == 'green'): # indicate correct button press
                                color = [50, 70]
                                screen.fill(BLACK) # Erase the work space
                                for text, text_pos in buttons_cali_t.items():
                                    text_surface = font_s.render(text, True, WHITE)
                                    rect = text_surface.get_rect(center=text_pos)
                                    screen.blit(text_surface, rect)
                                screen.blit(header_text, header_rect)
                                text_surface = font_m.render('GREEN', True, WHITE)
                                rect = text_surface.get_rect(center=(200, 225))
                                screen.blit(text_surface, rect)
                                text_surface = font_s.render('table color', True, WHITE)
                                rect = text_surface.get_rect(center=(200, 150))
                                screen.blit(text_surface, rect)
                                pygame.display.flip()
                            elif (text == 'done'): # indicate correct button press
                                cali_stage = 1
                                break
        screen.fill(BLACK) # Erase the work space
        header_text = font_l.render("Calibrate Stick", True, WHITE)
        header_rect = header_text.get_rect(center=(RES_X/2, 60))
        screen.blit(header_text, header_rect)
        cali_s_rects = {}
        for text, text_pos in buttons_cali_s.items():
            text_surface = font_s.render(text, True, WHITE)
            rect = text_surface.get_rect(center=text_pos)
            screen.blit(text_surface, rect)
            cali_s_rects[text] = rect # save rect for 'my-text' button
        text_surface = font_m.render(str(sens), True, WHITE)
        rect = text_surface.get_rect(center=(200, 225))
        screen.blit(text_surface, rect)
        text_surface = font_s.render('sensitivity', True, WHITE)
        rect = text_surface.get_rect(center=(200, 150))
        screen.blit(text_surface, rect)
        pygame.display.flip()
        while cali_stage == 1:      
            for event in pygame.event.get(): # for detecting an event for touch screen...
                if (event.type == MOUSEBUTTONUP):
                    pos = pygame.mouse.get_pos()
                    for (text, rect) in cali_s_rects.items(): # for saved button rects...
                        if (rect.collidepoint(pos)): # if collide with mouse click...
                            if (text == 'increase +'): # indicate correct button press
                                sens += 5
                                screen.fill(BLACK) # Erase the work space
                                text_surface = font_m.render(str(sens), True, WHITE)
                                rect = text_surface.get_rect(center=(200, 225))
                                screen.blit(text_surface, rect)
                                for text, text_pos in buttons_cali_s.items():
                                    text_surface = font_s.render(text, True, WHITE)
                                    rect = text_surface.get_rect(center=text_pos)
                                    screen.blit(text_surface, rect)
                                screen.blit(header_text, header_rect)
                                text_surface = font_s.render('sensitivity', True, WHITE)
                                rect = text_surface.get_rect(center=(200, 150))
                                screen.blit(text_surface, rect)
                                pygame.display.flip()
                            elif (text == 'decrease -'): # indicate correct button press
                                sens -= 5
                                screen.fill(BLACK) # Erase the work space
                                text_surface = font_m.render(str(sens), True, WHITE)
                                rect = text_surface.get_rect(center=(200, 225))
                                screen.blit(text_surface, rect)
                                for text, text_pos in buttons_cali_s.items():
                                    text_surface = font_s.render(text, True, WHITE)
                                    rect = text_surface.get_rect(center=text_pos)
                                    screen.blit(text_surface, rect)
                                screen.blit(header_text, header_rect)
                                text_surface = font_s.render('sensitivity', True, WHITE)
                                rect = text_surface.get_rect(center=(200, 150))
                                screen.blit(text_surface, rect)
                                pygame.display.flip()
                            elif (text == 'done'): # indicate correct button press
                                cali_stage = 0
                                break
        print('Calibrate Complete!')
        dictionary = {
            'color': color,
            'sensitivity': sens
        }

        json_object = json.dumps(dictionary, indent = 4)

        with open("cali.json", "w") as outfile:
            outfile.write(json_object)
        
        os.system('bash ./send_cali.sh')

        ui_run = True
        calibrate = False