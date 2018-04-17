import cv2 as cv

def get_bounding_boxes(img):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    return faces
