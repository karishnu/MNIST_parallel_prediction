import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def process_image():

    images2 = []

    img = cv2.imread('test_image.jpg', 0)
    im_gray = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    _, ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(c) for c in ctrs]

    max_w = 0
    max_h = 0

    print(rects)

    for rect in rects:
        (x, y, w, h) = rect

        if(w>max_w):
            max_w = w
        if(h>max_h):
            max_h = h

    if(max_h>=max_w):
        max_w = max_h
    else:
        max_h = max_w

    for rect in rects:
        (x, y, w, h) = rect

        if w >= 15 and h >= 30:
            image_cropped = im_th[y:y+h, x:x+w]
            img_w, img_h = image_cropped.shape
            background = np.array([[0]*max_w]*max_w, dtype = np.uint8)
            bg_w, bg_h = max_h, max_w
            offset = (int((bg_w - img_w) / 2), int((bg_h - img_h) / 2))
            background[offset[0]:offset[0]+img_w, offset[1]:offset[1]+img_h] = image_cropped
            resized_image = cv2.resize(background, (28, 28))
            images2.append(resized_image)

    # print(max_w, max_h)
    #
    # for i in range(len(images2)):
    #     plt.title(str(i))
    #     plt.subplot(2, 4, i + 1), plt.imshow(images2[i], 'gray')
    #     plt.xticks([]), plt.yticks([])
    #
    # plt.show()

    return images2

def main():
    process_image()

if __name__ == "__main__":
    main()

