#https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a

import numpy as np
import cv2
import time as t
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

cap = cv2.VideoCapture(0)
init_cal = False
x_c_1 = 0
y_c_1 = 0
x_c_2 = 0
y_c_2 = 0
x_pos = 0
prev_x = 0
prev_y = 0
lower_thresh_player = np.array([0, 0, 0])
upper_thresh_player = np.array([0, 0, 0])
counter = 0 

def track_player(frame, lower_thresh_player, upper_thresh_player):
    global x_pos
    global prev_x
    global prev_y
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
            print(prev_x)
            print(x)
            if abs(prev_x - x) <= 150:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                x_pos = x + int(w/2)
                prev_x = x + w//2
                prev_y = y + h//2
    #print( M )
    #cv.imshow('frame', frame)


def get_calibration_frames(frame):
    global x_c_1
    global x_c_2
    global y_c_1
    global y_c_2
    global counter
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
                print('top right corner calibration complete')
                print('hold led near left hip')
            if counter == 600:
                x_c_2 = x + (width//2)
                y_c_2 = x + (height//2)
                print(x_c_1)
                print(x_c_2)
                print(y_c_1)
                print(y_c_2)
                print('bottom left corner calibration complete')
    if counter == 600 and (x_c_2-x_c_1 <= 0 or x_c_1 == 0 or x_c_2 == 0 or y_c_2-y_c_1 <= 0 or y_c_1 == 0 or y_c_2 == 0):
        print('calibration failed...try again')
        counter = 0

def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items()))
    
    #for logging purposes
    print(perc)
    print(k_cluster.cluster_centers_)
    
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_): 
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)
        
    return palette


def make_histogram(cluster):
    """
    Count the number of pixels in each cluster
    :param: KMeans cluster
    :return: numpy histogram
    """
    numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=numLabels)
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist
    
def show_img_compar(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off')
    ax[1].axis('off')
    f.tight_layout()
    plt.show()

def make_bar(height, width, color):
    """
    Create an image of a given color
    :param: height of the image
    :param: width of the image
    :param: BGR pixel values of the color
    :return: tuple of bar, rgb values, and hsv values
    """
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv_bar[0][0]
    return bar, (red, green, blue), (hue, sat, val)

def calibrate(frame, x_c_1, y_c_1, x_c_2, y_c_2):
    global lower_thresh_player
    global upper_thresh_player
    global prev_x
    global prev_y
    global x_pos
    cv2.rectangle(frame, (x_c_1, y_c_1), (x_c_2, y_c_2), (0, 255, 0), 3)
    calibration_frame = frame[y_c_1:y_c_2, x_c_1:x_c_2]
    cv2.imshow('calibration frame', calibration_frame)
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
    lower_thresh_player = np.array([int(avg_h)-30,int(avg_s)-40,int(avg_v)-40])
    upper_thresh_player = np.array([int(avg_h)+30,int(avg_s)+100,int(avg_v)+100])
    prev_x = (x_c_2 + x_c_1)//2
    prev_y = (y_c_2 + y_c_1)//2
    clt= KMeans(n_clusters=3)
    clt.fit(calibration_frame.reshape(-1, 3))
    show_img_compar(calibration_frame, palette_perc(clt))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if counter <= 600:
        get_calibration_frames(frame)
    elif counter == 601:
        print('center shirt in box and stay still')
    elif counter == 602:
        t.sleep(3)
        calibrate(frame, x_c_1, y_c_1, x_c_2, y_c_2)
    else:
        track_player(frame, lower_thresh_player, upper_thresh_player)
        #print(x_pos)


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
