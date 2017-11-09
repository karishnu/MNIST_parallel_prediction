import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

import matplotlib.pyplot as plt

im = cv2.imread('test_image.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,121,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Detect contours using both methods on the same image
_, contours1, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# Copy over the original image to separate variables
img1 = im.copy()

for c in contours1:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect

    if(w<100 and h<100 and h>40):
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(img1[x:x+y, y:y+h]/255)

cv2.imshow("Show", img1)
cv2.waitKey()