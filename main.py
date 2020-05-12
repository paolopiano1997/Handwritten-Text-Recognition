from LineSegmentation import LineSegmentation
import cv2
import numpy as np
from scan import scanfun
import sys
from inference import predict
import ntpath
import os

def line_segment(image, img, output_path):
    """
    img: image needed to segment to lines.
    output_path: output folder of segmented images. 
    :return: output image pathes
    """
    #cv2.imshow("Original", img)
    #cv2.waitKey()
    # (1) Segment Lines
    line_segmentation = LineSegmentation(img=img, output_path=output_path)
    lines = line_segmentation.segment()
    # (2) Save lines to file
    output_image_path = line_segmentation.save_lines_to_file(lines)
    percTot=0
    count=0
    text=""
    for m in lines:
        pd,pbperc = predict(cv2.cvtColor(m,cv2.COLOR_GRAY2RGB))
        #cv2.imshow("Line",m)
        #cv2.waitKey()
        percTot+=pbperc
        count+=1
        text+=pd +'\n'
    if count!=0:
        perc=percTot/count
    if count==0:
        pd,pbperc = predict(cv2.cvtColor(cv2.imread(image),cv2.COLOR_GRAY2RGB))
        perc = pbperc
        text=pd
    head, tail = ntpath.split(image)
    image =  tail or ntpath.basename(head)
    filename = os.path.splitext(image)[0]
    fout = open('output/' + filename + '_output.txt','w')
    fperc = open('output/' + filename + '_probability.txt','w')
    fout.write(text)
    fperc.write(f"{perc:.2f}")
    
def start(image):

    #scan the image and it will be saved as out.png
    scanfun(image)

    #read the image to resize and crop lines
    img=cv2.imread("test/out.png")
    #print(img)

    #percent by which the image is resized
    scale_percent = 50

    #calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(img, dsize)
    cv2.imwrite('test/out2.png',output)

    #do program
    output_path = "output"
    line_segment(image,output, output_path)



