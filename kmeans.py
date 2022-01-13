#https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
#https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#changed the image to be video capture
#added a while loop around the code generating the histogram from frame of video
#used plt.ion so that it uses video strem instead of individual frames
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar
    
cap = cv2.VideoCapture(0)
while(1):
    _, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.imshow(bar)
    plt.ion()
    plt.show()
    cv2.imshow('frame',frame)
    k = cv2.waitKey(5) & 0xFF
    if k ==27:
        break
cv2.destroyAlllWindows()
