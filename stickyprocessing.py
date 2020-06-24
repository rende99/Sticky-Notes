from skimage import data
from skimage.feature import canny # helps detect edges in images
import numpy as np
import pandas as pd
import matplotlib
import cv2 as cv
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from skimage.exposure import histogram
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed
import skimage.color as color
from skimage.color import label2rgb
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


img = cv.imread('sticky.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

f = color.rgb2gray(img)
imgASNP = np.asarray(f, dtype='int32')

elevation_map = sobel(imgASNP)
fill_coins = ndi.binary_fill_holes(elevation_map)


labeled_coins, _ = ndi.label(fill_coins)
image_label_overlay = label2rgb(labeled_coins, image=img, bg_label=0)
plt.imshow(image_label_overlay)
plt.show()
print(image_label_overlay)
obs = ndi.find_objects(labeled_coins)


for ob in range(len(obs)):
    slice_x, slice_y = ndi.find_objects(labeled_coins)[ob]
    print(slice_x, slice_y)
    roi = img[slice_x, slice_y]
    print(roi.size)
    #make sure the image is a proper size, and not just a visual error.
    if roi.size > 1000:
        plt.imshow(roi)
        stickyText = pytesseract.image_to_string(roi)
        stickyText = stickyText.rstrip()
        print(stickyText)
        plt.savefig('output/sticky_'+ str(ob) +'.png')
        f = open('output/sticky_'+ str(ob) +'_content.txt',"w+")
        f.write(stickyText)
        plt.show()

