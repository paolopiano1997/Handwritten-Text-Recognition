# import required libraries
import numpy as np
import cv2
import imutils

def scanfun(name):
    
    # read the image
    image = cv2.imread(name)
    orig = image.copy()


    # show the original image
    #cv2.imshow("Original Image", image)
    #cv2.waitKey(0) # press 0 to close all cv2 windows
    #cv2.destroyAllWindows()

    # convert image to gray scale. This will remove any color noise
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ------------------------------
    # convert to black/white with high contrast for documents

    from skimage.filters import threshold_local

    # increase contrast incase its document
    T = threshold_local(grayImage, 9, offset=8, method="gaussian")
    scanBW = (grayImage > T).astype("uint8") * 255

    # display final high-contrast image
    #cv2.imshow("scanBW", scanBW)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite("test/out.png", scanBW)
