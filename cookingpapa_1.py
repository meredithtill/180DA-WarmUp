import pygame
from pygame.locals import *
import numpy as np
import cv2
import time as t
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pygame.init()

clock = pygame.time.Clock()
fps = 60

screen_width = 1200
screen_height = 900

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Cooking Papa 1.0')

#define game variables
tile_size = 50

atStove = False
atBoard = False

cap = cv2.VideoCapture(0)
init_cal = False
x_c_1 = 0
y_c_1 = 0
x_c_2 = 0
y_c_2 = 0
lower_thresh_player = np.array([0, 0, 0])
upper_thresh_player = np.array([0, 0, 0])
counter = 0 
x_pos  = 0

def track_player(frame, lower_thresh_player, upper_thresh_player):
	global x_pos
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# define range of blue color in HSV
	# define range of blue color in HSV
	lower_blue = np.array([90,50,50])
	upper_blue = np.array([255,255,255])
	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, lower_thresh_player, upper_thresh_player)
	ret,thresh = cv2.threshold(mask,127,255,0)
	res = cv2.bitwise_and(frame,frame, mask= mask)
	#from threshholding cv doc
	th3 = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for i in contours:
		area = cv2.contourArea(i)
		if area > 4000:
			x,y,w,h = cv2.boundingRect(i)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			x_pos = x + int(w/2)
	#print( M )
	#cv.imshow('frame', frame)


def get_calibration_frames(frame):
	global x_c_1
	global x_c_2
	global y_c_1
	global y_c_2
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_thresh_LED = np.array([0, 0, 250]) #LED
	upper_thresh_LED = np.array([179, 10, 255]) #LED

	mask = cv2.inRange(hsv, lower_thresh_LED, upper_thresh_LED)
	_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if counter == 1:
		print('stand in middle of frame and remain still during calibration')
	if counter == 25:
		print('hold led near top of right shoulder')
	for i in contours:
		#get rid of noise first by calculating area
		area = cv2.contourArea(i)
		if area > 100 and area < 400:
			#cv2.drawContours(frame, [i], -1, (0, 255, 0), 2)
			x, y, width, height = cv2.boundingRect(i)
			cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
			x2 = x + width
			y2 = y + height
			if counter == 300:
				x_c_1 = x + (width//2)
				y_c_1 = x + (height//2)
				print('bottom left corner calibration complete')
				print('hold led near left hip')
			if counter == 575:
				x_c_2 = x + (width//2)
				y_c_2 = x + (height//2)
				print(x_c_1)
				print(x_c_2)
				print(y_c_1)
				print(y_c_2)
				print('top right corner calibration complete')

def calibrate(frame, x_c_1, y_c_1, x_c_2, y_c_2):
	global lower_thresh_player
	global upper_thresh_player
	global x_pos
	cv2.rectangle(frame, (x_c_1, y_c_1), (x_c_2, y_c_2), (0, 255, 0), 3)
	calibration_frame = frame[y_c_1:y_c_2, x_c_1:x_c_2]
	cal_hsv = cv2.cvtColor(calibration_frame, cv2.COLOR_BGR2HSV)
	x_pos = int(abs(x_c_1 - x_c_2)/2)
	h_val = cal_hsv[:,:,0]
	s_val = cal_hsv[:,:,1]
	v_val = cal_hsv[:,:,2]
	h_val.sort()
	s_val.sort()
	v_val.sort()
	#discard outliers
	(h,w) = h_val.shape
	h_low = h//8
	w_low = w//8
	h_high = h-h_low
	w_high = w-w_low
	h_val_ab = h_val[h_low:h_high,w_low:w_high]
	s_val_ab = s_val[h_low:h_high,w_low:w_high]
	v_val_ab = v_val[h_low:h_high,w_low:w_high]
	avg_h = np.average(h_val_ab)
	avg_s = np.average(s_val_ab)
	avg_v = np.average(v_val_ab)
	hsv_avg = np.array([int(avg_h),int(avg_s),int(avg_v)])
	lower_thresh_player = np.array([int(avg_h)-30,int(avg_s)-20,int(avg_v)-20])
	upper_thresh_player = np.array([int(avg_h)+30,int(avg_s)+100,int(avg_v)+100])

#load images
#sun_img = pygame.image.load('img/sun.png')
bg_img = pygame.image.load('img/background.png')
bg_img = pygame.transform.scale(bg_img, (1200, 900))
bg_chopping = pygame.image.load('img/chopping.png')
bg_chopping = pygame.transform.scale(bg_chopping, (1200, 900))
bg_stove = pygame.image.load('img/stove.png')
bg_stove = pygame.transform.scale(bg_stove, (1200, 900))

class Player():
	def __init__(self, x, y):
		self.images_right = []
		self.images_left = []
		self.index = 0
		self.counter = 0
		for num in range(1, 5):
			img_right = pygame.image.load(f'img/chef{num}.png')
			img_right = pygame.transform.scale(img_right, (300, 600))
			img_left = pygame.transform.flip(img_right, True, False)
			self.images_right.append(img_right)
			self.images_left.append(img_left)
		self.image = self.images_right[self.index]
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y
		self.vel_y = 0
		self.jumped = False
		self.direction = 0

	def isAtBoard(self):
		if self.rect.x > screen_width/2 and self.rect.x > 200:
			return True
		else:
			return False


	def update(self):
		dx = 0
		dy = 0
		walk_cooldown = 5

		#get keypresses
		#this moves the plauyer
		key = pygame.key.get_pressed()
		if key[pygame.K_SPACE] and self.jumped == False:
			self.vel_y = -15
			self.jumped = True
		if key[pygame.K_SPACE] == False:
			self.jumped = False
		#if key[pygame.K_LEFT]:
		if (self.rect.x - x_pos) < 0:
			#print(dx)
			#print(self.rect.x)
			#print(x_pos)
			#dx -= abs(self.rect.x - x_pos)
			#print(dx)
			self.counter += 1
			self.direction = -1
		#if key[pygame.K_RIGHT]:
		if (self.rect.x - x_pos) > 0:
			#dx += abs(self.rect.x - x_pos)
			self.counter += 1
			self.direction = 1
		if key[pygame.K_LEFT] == False and key[pygame.K_RIGHT] == False:
			self.counter = 0
			self.index = 0
			if self.direction == 1:
				self.image = self.images_right[self.index]
			if self.direction == -1:
				self.image = self.images_left[self.index]


		#handle animation
		if self.counter > walk_cooldown:
			self.counter = 0	
			self.index += 1
			if self.index >= len(self.images_right):
				self.index = 0
			if self.direction == 1:
				self.image = self.images_right[self.index]
			if self.direction == -1:
				self.image = self.images_left[self.index]


		#add gravity
		self.vel_y += 1
		if self.vel_y > 10:
			self.vel_y = 10
		dy += self.vel_y

		#check for collision

		#update player 
		print(dx)
		self.rect.x = x_pos
		self.rect.y += dy

		if self.rect.bottom > screen_height:
			self.rect.bottom = screen_height
			dy = 0

		#draw player onto screen
		screen.blit(self.image, self.rect)


'''ignore the World Class'''

class World():
	def __init__(self, data):
		self.tile_list = []

		#load images
		dirt_img = pygame.image.load('img/dirt.png')
		grass_img = pygame.image.load('img/grass.png')

		row_count = 0
		for row in data:
			col_count = 0
			for tile in row:
				if tile == 1:
					img = pygame.transform.scale(dirt_img, (tile_size, tile_size))
					img_rect = img.get_rect()
					img_rect.x = col_count * tile_size
					img_rect.y = row_count * tile_size
					tile = (img, img_rect)
					self.tile_list.append(tile)
				if tile == 2:
					img = pygame.transform.scale(grass_img, (tile_size, tile_size))
					img_rect = img.get_rect()
					img_rect.x = col_count * tile_size
					img_rect.y = row_count * tile_size
					tile = (img, img_rect)
					self.tile_list.append(tile)
				col_count += 1
			row_count += 1

	def draw(self):
		for tile in self.tile_list:
			screen.blit(tile[0], tile[1])



world_data =[
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]


'''MAIN Program'''
inWorld = True

player = Player(100, screen_height - 130)
world = World(world_data)


run = True
while run:
	clock.tick(fps)

	# Capture frame-by-frame
	ret, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	if counter <= 600:
		get_calibration_frames(frame)
	elif counter == 601:
		print('in pass statement')
		pass
	elif counter == 602:
		print('calibrating...')
		t.sleep(3)
		calibrate(frame, x_c_1, y_c_1, x_c_2, y_c_2)
	elif counter > 602:
		track_player(frame, lower_thresh_player, upper_thresh_player)
		if player.isAtBoard() == False:
			atStove = True
			atBoard = False
		else:
			atBoard = True
			atStove = False
		keys_pressed = pygame.key.get_pressed()
		if keys_pressed[pygame.K_UP]:
			inWorld=False
		if keys_pressed[pygame.K_DOWN]:
			inWorld=True
		#if keys_pressed[pygame.K_a]: #a to go left
		#	inWorld = False
		        

		if (inWorld):
			screen.blit(bg_img, (0, 0))
			world.draw()
			player.update()
		elif atBoard == True:
			screen.blit(bg_chopping, (0, 0))
		else:
			screen.blit(bg_stove, (0, 0))


		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

		pygame.display.update()


	counter = counter+1
	cv2.imshow('calibrating frame', frame)
	cv2.resizeWindow('calibrating frame', 600,600)
	#print(counter)
	# Display the resulting frames
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	
#screen.blit(sun_img, (100, 100))

pygame.quit()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()