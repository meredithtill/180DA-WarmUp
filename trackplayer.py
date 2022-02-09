#https://www.pyimagesearch.com/2021/01/20/opencv-getting-and-setting-pixels/

import numpy as np
import cv2
import time as t
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(0)
init_cal = False
x_c_1 = 0
y_c_1 = 0
x_c_2 = 0
y_c_2 = 0
lower_thresh_player = np.array([0, 0, 0])
upper_thresh_player = np.array([0, 0, 0])
counter = 0 

def track_player(frame, lower_thresh_player, upper_thresh_player):
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
        if area > 700:
            x,y,w,h = cv2.boundingRect(i)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
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
        print('hold led near top of left shoulder')
    for i in contours:
        #get rid of noise first by calculating area
        area = cv2.contourArea(i)
        if area > 100 and area < 400:
            #cv2.drawContours(frame, [i], -1, (0, 255, 0), 2)
            x, y, width, height = cv2.boundingRect(i)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
            x2 = x + width
            y2 = y + height
            if counter == 400:
                x_c_1 = x + (width//2)
                y_c_1 = x + (height//2)
                print('bottom left corner calibration complete')
                print('hold led near left hip')
            if counter == 800:
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
    cv2.rectangle(frame, (x_c_1, y_c_1), (x_c_2, y_c_2), (0, 255, 0), 3)
    print(x_c_1)
    print(x_c_2)
    print(y_c_1)
    print(y_c_2)
    calibration_frame = frame[y_c_1:y_c_2, x_c_1:x_c_2]
    cal_hsv = cv2.cvtColor(calibration_frame, cv2.COLOR_BGR2HSV)
    print(cal_hsv.shape)
    h_val = cal_hsv[:,:,0]
    s_val = cal_hsv[:,:,1]
    v_val = cal_hsv[:,:,2]
    h_val.sort()
    s_val.sort()
    v_val.sort()
    #discard outliers
    print(h_val.shape)
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
    print(hsv_avg)
    lower_thresh_player = np.array([int(avg_h)-10,int(avg_s)-10,int(avg_v)-10])
    upper_thresh_player = np.array([int(avg_h)+10,255,255])
    print(lower_thresh_player)
    print(upper_thresh_player)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if counter <= 800:
        get_calibration_frames(frame)
    elif counter == 801:
        calibrate(frame, x_c_1, y_c_1, x_c_2, y_c_2)
    elif counter > 801 and counter < 830:
        print('center shirt in box and stay still')
        cv2.rectangle(frame, (x_c_1, y_c_1), (x_c_2, y_c_2), (0, 255, 0), 3)
    else:
    	track_player(frame, lower_thresh_player, upper_thresh_player)


    counter = counter+1
    cv2.imshow('calibrating frame', frame)
    cv2.resizeWindow('calibrating frame', 600,600)
    #print(counter)
    # Display the resulting frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()