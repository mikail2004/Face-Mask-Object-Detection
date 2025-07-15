# Mikail Usman 
# FaceMask Object Detection

import os
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2

np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/Users/m20mi/Documents/Work/FaceMask/classifier_model.h5", compile=False)

# Load the labels
class_names = ['Wearing Mask', 'No Mask']

def classifyImage(image):
    # Processing image
    #image = cv2.imread(imagePath, cv2.IMREAD_COLOR) # Use OpenCV to read image from specified path
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA) # Resize the raw image into (224-height,224-width) pixels
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3) # Make the image a numpy array and reshape it to the models input shape.
    image = image / 255.0 # Normalize the image array

    # Model predicts class for the given image
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Return prediction and confidence score
    predScore = str(np.round(confidence_score * 100))[:-2] + '%'

    return class_name, predScore

if __name__ == '__main__':
    img1 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/50.png"
    img2 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/582.png"
    img3 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/5853.png"
    img4 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/4686.png"
    img5 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/0922.png"
    img6 = "C:/Users/m20mi/Documents/Work/FaceMask/Eval_Images/5992.jpg"

    img6 = cv2.imread(img6, cv2.IMREAD_COLOR) # Use OpenCV to read image from specified path
    img6 = cv2.resize(img6, (500, 600), interpolation=cv2.INTER_AREA) # Resize the raw image into (224-height,224-width) pixels

    # Load Haar Cascade XML file (data for which object you want to detect)
    faceCascade = cv2.CascadeClassifier("C:/Users/m20mi/Documents/Work/FaceMask/haarcascade_frontalface_default.xml")

    # Detects mutliple objects in a given image. 
    # <minSize> as drop off threshold for detected object size
    faceDetected = faceCascade.detectMultiScale(img6, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Obtain bounding box coordinates (detected object) and draw them
    for (x, y, w, h) in faceDetected:
        # Draws a rectangle on the image at the coordinates (x, y) with a width w and height h. 
        # The color is green (0, 255, 0) and the rectangle has a thickness of 5 pixels.
        cv2.rectangle(img6, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # Crops detected face from image
        face = img6[y:y+h, x:x+w]

        # Run classifier model on cropped face
        face_resized = cv2.resize(face, (224, 224)) / 255.0
        face_input = np.expand_dims(face_resized, axis=0)
        label = class_names[np.argmax(model.predict(face_input))]
        cv2.putText(img6, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Output Image", img6) # Show the image in a window

    #print(classifyImage(img1))
    cv2.waitKey(0) # Keep image window open until user closes it
    cv2.destroyAllWindows() # Delete all windows from memory once user closes them