import cv2 as cv

def get_bounding_boxes(file_path_and_name):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv.imread(file_path_and_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    return faces
