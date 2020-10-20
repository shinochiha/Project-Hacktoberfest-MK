import tornado.web
import time
import os
import cv2
import sys

def faceDetection(fileName):

    # Get user supplied values
    imagePath = f"images/{fileName}"
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    print("Faces!", faces)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (90, 59, 235), 2)
        # roi_color = image[y:y + h, x:x + w]
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        # ms = time.time_ns()
        # cv2.imwrite(f'{timestamp}{ms}.jpg', roi_color)

    return faces
    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)
