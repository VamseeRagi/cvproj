import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
img = cv.imread('shield_cast.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.2, 3)
print(faces)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()

#%%
#Note: For now, we are only considering groups of 10 faces of size 128x128
#Rosie is taking care of the resizing part.

n = 10

headshots = np.zeros((128,128,3*n))

for i in range(0,faces.shape[0]):
    j = 3*i
    box = faces[i,:]
    pic = img[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2]),:]
    headshots[:,:,j:(j+3)] = pic

#feed headshots into model built by Kye

#%%
#assume p is the output of the model of size 10x1 containing the probabilities
#of the suspect matching the candidate

p = np.array([1, 3, 5, 2])
i = p.argmax()
box = faces[i,:]
img = cv.imread('shield_cast.jpg')
cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()