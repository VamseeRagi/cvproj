import cv2 as cv
import numpy as npsh
import scipy.io as sc

from detection import get_bounding_boxes

candidate = input("Candidate File Name: ")
suspect = input("Suspect File Name: ")

img1 = cv.imread(candidate,1)
img2 = cv.imread(suspect,1)

faces1 = get_bounding_boxes(img1)
faces2 = get_bounding_boxes(img2)

print(faces1.shape)
print(faces2.shape)
#%%

n = 10

X = np.zeros((n+1,100,100,3))

for i in range(0,min(n,faces1.shape[0])+1):
    if i == 0:
        box = faces2[1,:]
        pic = img2[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2]),:]
        newPic = None
        dimx = pic.shape[0]
        dimy = pic.shape[1]
        newPic = cv.resize(pic, newPic, fx=100/dimx, fy=100/dimy)
        X[i, :, : ,:] = newPic
    else:
        box = faces1[i-1,:]
        pic = img1[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2]),:]
        newPic = None
        dimx = pic.shape[0]
        dimy = pic.shape[1]
        newPic = cv.resize(pic, newPic, fx=100/dimx, fy=100/dimy)
        X[i, :, : ,:] = newPic

#%%

##Model Stuff

#%%

i = p.argmax()
box = faces[i,:]
img = cv.imread('shield_cast.jpg')
cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()


cv.imshow('image',img1)
cv.waitKey(0)
cv.destroyAllWindows()