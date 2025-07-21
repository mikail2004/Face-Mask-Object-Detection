# Mikail Usman 
# Face Mask Object Detection

import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2

np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/Users/m20mi/Documents/Work/FaceMask/classifier_model.h5", compile=False)
class_names = ['Wearing Mask', 'No Mask']

# Process image by resizing, converting to array
def processImage(inputIMG):
    imgRead = cv2.imread(inputIMG, cv2.IMREAD_COLOR) # Use OpenCV to read image from specified path
    imgResize = cv2.resize(imgRead, (500, 500), interpolation=cv2.INTER_AREA) # Resize the raw image with square dimensions matching training data dimensions

    return imgResize

# Object detection using classic CV (Haar Cascades)
def haarFaceDetection(inputIMG):
    # Load Haar Cascade XML file (data for which object you want to detect)
    faceCascade = cv2.CascadeClassifier("C:/Users/m20mi/Documents/Work/FaceMask/haarcascade_frontalface_default.xml")

    # Detects mutliple objects in a given image. 
    # <minSize> as drop off threshold for detected object size
    faceDetected = faceCascade.detectMultiScale(inputIMG, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    return faceDetected

def drawBoundingBox(inputImg):
    # Obtain bounding box coordinates (detected object) and draw them
    for (x, y, w, h) in haarFaceDetection(inputImg):
        # Draws a rectangle on the image at the coordinates (x, y) with a width w and height h. 
        # The color is green (0, 255, 0) and the rectangle has a thickness of 5 pixels.
        cv2.rectangle(inputImg, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # Crops detected face from image
        face = inputImg[y:y+h, x:x+w]

        # Run classifier model on cropped face
        face_resized = cv2.resize(face, (224, 224)) / 255.0
        face_input = np.expand_dims(face_resized, axis=0)
        label = class_names[np.argmax(model.predict(face_input))]
        cv2.putText(inputImg, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return inputImg

def liveDetection():
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        cv2.imshow('FaceMask', drawBoundingBox(frame))

        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img1 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/50.png"
    img2 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/582.png"
    img3 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/5853.png"
    img4 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/111.png"
    img5 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/222.jpg"

    liveDetection()

    # selectedImg = processImage(img5)
    # processedImg = drawBoundingBox(selectedImg)
    # cv2.imshow("Output Image", processedImg)

    # #print(classifyImage(img1))
    # cv2.waitKey(0) # Keep image window open until user closes it
    # cv2.destroyAllWindows() # Delete all windows from memory once user closes them