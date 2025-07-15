# Mikail Usman 
# FaceMask Object Detection

import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2

np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/Users/m20mi/Documents/Work/FaceMask/classifier_model.h5", compile=False)
class_names = ['Wearing Mask', 'No Mask']

if __name__ == '__main__':
    img1 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/50.png"
    img2 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/582.png"
    img3 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/5853.png"
    img4 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/4686.png"
    img5 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/0922.png"
    img6 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/111.png"

    selectedImg = cv2.imread(img6, cv2.IMREAD_COLOR) # Use OpenCV to read image from specified path
    selectedImg = cv2.resize(selectedImg, (500, 600), interpolation=cv2.INTER_AREA) # Resize the raw image into (224-height,224-width) pixels

    # Load Haar Cascade XML file (data for which object you want to detect)
    faceCascade = cv2.CascadeClassifier("C:/Users/m20mi/Documents/Work/FaceMask/haarcascade_frontalface_default.xml")

    # Detects mutliple objects in a given image. 
    # <minSize> as drop off threshold for detected object size
    faceDetected = faceCascade.detectMultiScale(selectedImg, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    # Obtain bounding box coordinates (detected object) and draw them
    for (x, y, w, h) in faceDetected:
        # Draws a rectangle on the image at the coordinates (x, y) with a width w and height h. 
        # The color is green (0, 255, 0) and the rectangle has a thickness of 5 pixels.
        cv2.rectangle(selectedImg, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # Crops detected face from image
        face = selectedImg[y:y+h, x:x+w]

        # Run classifier model on cropped face
        face_resized = cv2.resize(face, (224, 224)) / 255.0
        face_input = np.expand_dims(face_resized, axis=0)
        label = class_names[np.argmax(model.predict(face_input))]
        cv2.putText(selectedImg, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Output Image", selectedImg) # Show the image in a window

    #print(classifyImage(img1))
    cv2.waitKey(0) # Keep image window open until user closes it
    cv2.destroyAllWindows() # Delete all windows from memory once user closes them