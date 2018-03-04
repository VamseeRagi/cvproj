import cv2 as cv
from matplotlib import pyplot as plt

import numpy as np


## load dog picture, show it, and store the greyscale image

img = cv.imread('dog.jpg',0)
cv.imshow('image',img)
k = cv.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv.imwrite('dog_grey.png',img)
    cv.destroyAllWindows()

# alternatively, can use matplotlib to show
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
