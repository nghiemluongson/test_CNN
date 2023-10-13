import numpy as np
import cv2
import pickle

################################
frame_Width = 640
frame_Height = 480
brightness = 180
threshold = 0.90
font = cv2.FONT_HERSHEY_SIMPLEX
################################

### setup video camera ###
cap = cv2.VideoCapture(0)
cap.set(3, frame_Width)
cap.set(4, frame_Height)
cap.set(10, brightness)
################################

### import the trained model ###
from tensorflow import keras
my_Model = keras.models.load_model('my model')
my_Model.load_weights("weights.h5")

# pickle_In = open("keras_model.h5", "rb")
# model = pickle.load(pickle_In)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def get_Class_Name(class_No):
    if class_No == 0:
        return "Red"
    elif class_No == 1:
        return "Yellow"
    elif class_No == 2:
        return "Green"

while True:
    success, imgOrignal = cap.read()

    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "Class: ", (20, 35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "Probability: ", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    predictions = model.predict(img)
    class_Index = model.predict_classes(img)
    probability_Value = np.amax(predictions)

    if probability_Value > threshold:
        cv2.putText(imgOrignal, str(class_Index) + " " + str(get_Class_Name(class_Index)), (120,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probability_Value*100, 2)) + '%', (180, 35), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


