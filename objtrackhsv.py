#sources used
#https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
#added a for loop and checked area
#https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
#https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga95f5b48d01abc7c2e0732db24689837b

import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    # define range of blue color in HSV
    lower_blue = np.array([90,50,50])
    upper_blue = np.array([255,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    ret,thresh = cv.threshold(mask,127,255,0)
    res = cv.bitwise_and(frame,frame, mask= mask)
    #from threshholding cv doc
    th3 = cv.adaptiveThreshold(mask,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i in contours:
        area = cv.contourArea(i)
        if area > 5000:
            x,y,w,h = cv.boundingRect(i)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #print( M )
    cv.imshow('frame', frame)
    #cv.imshow('mask', mask)
    #cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
