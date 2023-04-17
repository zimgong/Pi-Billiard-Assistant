import cv2 as cv
import imutils
import pygame

#OBTAIN AND PRE-PROCESS POOL TABLE IMAGE
image = cv.imread('coins.jpg', cv.IMREAD_COLOR)
imagebackground = cv.imread('background.jpg', cv.IMREAD_COLOR)
imagecpy = image
#image = image-imagebackground

#convert to grayscale, equalize histogram and blur
grayImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
grayBG = cv.cvtColor(imagebackground, cv.COLOR_BGR2GRAY)
grayImg = grayImg-grayBG
grayImg = cv.equalizeHist(grayImg)
grayImg = cv.medianBlur(grayImg,5)

#PYGAME INIT, DISPLAY INIT
pygame.init()

WHITE = 255, 255, 255
my_font = pygame.font.Font(None, 25)
screen=pygame.display.set_mode((len(imagecpy), len(imagecpy[0])))

class ball:
    def __init__(self, centerx, centery, ballspeedx, ballspeedy):
        self.center = (centerx, centery)
        self.ballspeed = (ballspeedx, ballspeedy)

cuePresent = False

#CIRCLE AND CUE DETECTION
circles= cv.findContours(grayImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#param1 = grayImg.shape[0]/12
#circles = cv.HoughCircles(grayImg, cv.HOUGH_GRADIENT, 1, param1,200,449,2,7)
circles = imutils.grab_contours(circles)
if(circles is not None):
    #circles = numpy.uint16(numpy.around(circles))
    ballArray = []
    for i in circles:
        #if(i[2]<12 and i[2]>0):
        ((x,y), radius) = cv.minEnclosingCircle(i)
        center = (int(x),int(y))
        text_surface = my_font.render('c', True, WHITE)
        rect = text_surface.get_rect(center=center)
        screen.blit(text_surface, rect)
    
    pygame.display.flip()