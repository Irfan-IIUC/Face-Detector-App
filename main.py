'''
========== Face Detector App ==========
step-1/ get a crap-load of faces [tons of faces]
step-2/ make them all black and white [gray-scale]
step-3/ train the algorithm to detect faces
'''

# pip install opencv-python
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

import cv2
from random import randrange

# load some pre-trained data on face frontal from open-cv [haarcascade algorithm]
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
# img = cv2.imread('tamim_mushfiq.png')

# to capture video from webcam [0 means default webcam else give video path location]
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:

    # read the current frame
    successful_frame_read, frame = webcam.read()

    # must convert to gray scale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

    # show video
    cv2.imshow('Face Detection App', frame)
    key = cv2.waitKey(1)

    # press Q or q to quit
    if key == 81 or key == 113:
        break

'''
========== first try it for image ==========
# must convert to gray scale
grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 3)

# show image
cv2.imshow('Mushfiqur Rahim', img)
cv2.waitKey()
'''

print('code completed')
