#https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
#https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
#https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga95f5b48d01abc7c2e0732db24689837b
#added for loop and checked area
import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    _, frame = cap.read()
    lower_blue = np.array([140,0,0])
    upper_blue = np.array([255,255,120])
    mask = cv.inRange(frame, lower_blue, upper_blue)
    ret, thresh = cv.threshold(mask,127,255,0)
    res = cv.bitwise_and(frame,frame,mask=mask)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i in contours:
        area = cv.contourArea(i)
        if area > 5000:
            x,y,w,h = cv.boundingRect(i)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imshow('frame', frame)
    k = cv.waitKey(5) & 0xFF
    if k ==27:
        break
cv.destroyAllWindows()
