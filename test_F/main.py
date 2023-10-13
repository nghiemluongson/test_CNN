import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

########## Parameters ##########
path = "myData"
label_File = "labels.csv"
batch_Size_Val = 50
steps_Per_Epoch_Val = 502
epochs_Val = 10
image_Dimesions = (32,32,3)
test_Ratio = 0.1
validation_Ratio = 0.2
################################

# -----------------------------------------------------------

########## Importing the images ##########
count = 0
images = []
class_No = []
myList = os.listdir(path)
print("Total classes detected:", len(myList))
no_Of_Classes = len(myList)
print("Importing classes ...")

for x in range(0, len(myList)):
    my_Pic_List = os.listdir(path + '/' + str(count))
    for y in my_Pic_List:
        cur_Img = cv2.imread(path + '/' + str(count) + '/' + y)
        images.append(cur_Img)
        class_No.append(count)
    print(count, end=" ")
    count += 1

print(" ")
images = np.array(images)
class_No = np.array(class_No)
################################

# -----------------------------------------------------------

########## Split data ##########
X_train, X_test, y_train, y_test = train_test_split(images, class_No, test_size=test_Ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_Ratio)
################################

# -----------------------------------------------------------

########## Check if number of images matches to number of labels for each data set ##########
print("Data shapes")
print("Train", end=""); print(X_train.shape, y_train.shape)
print("Validation", end=""); print(X_validation.shape, y_validation.shape)
print("Test", end=""); print(X_test.shape, y_test.shape)

assert(X_train.shape[0]==y_train.shape[0])
assert(X_validation.shape[0]==y_validation.shape[0])
assert(X_test.shape[0]==y_test.shape[0])
assert(X_train.shape[1:]==(image_Dimesions))
assert(X_validation.shape[1:]==(image_Dimesions))
assert(X_test.shape[1:]==(image_Dimesions))
###############################

# -----------------------------------------------------------

########## Read csv file ##########
data = pd.read_csv(label_File)
print("Data shape: ", data.shape, type(data))
################################

# -----------------------------------------------------------

########## Displays some examples images of all the classes ##########
num_Of_Samples = []
cols = 5
num_Classes = no_Of_Classes
fig, axs = plt.subplots(nrows=num_Classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        X_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + row["Name"])
            num_Of_Samples.append(len(x_selected))
################################

# -----------------------------------------------------------

########## Displays a bar chart showing no of samples for each category ##########
print(num_Of_Samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_Classes), num_Of_Samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()
###############################

# -----------------------------------------------------------

########## Pre-processing the images ##########
def grayscales(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscales(img)
    img = equalize(img)
    img = img / 255
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
cv2.imshow("Grayscale Images", X_train[random.randint(0, len(X_train) - 1)])

###############################

# -----------------------------------------------------------

########## Add a depth of 1 ##########
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
###############################

# -----------------------------------------------------------

########## Argument of images: to make it more genetic ##########
data_Gen = ImageDataGenerator(width_shift_range=0.1,
                              height_shift_range=0.1,
                              zoom_range=0.2,
                              shear_range=0.1,
                              rotation_range=10)
data_Gen.fit(X_train)
batches = data_Gen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(image_Dimesions[0], image_Dimesions[1]))
    axs[i].axis("off")
plt.show()

y_train = to_categorical(y_train, no_Of_Classes)
y_validation = to_categorical(y_validation, no_Of_Classes)
y_test = to_categorical(y_test, no_Of_Classes)
################################

# -----------------------------------------------------------

########## CNN ##########
def myModel():
    no_Of_Filters = 60
    size_Of_Filter = (5, 5)
    size_Of_Filter2 = (3, 3)
    size_Of_Pool = (2, 2)
    no_Of_Nodes = 500

    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_Of_Filter, input_shape=(image_Dimesions[0], image_Dimesions[1], 1), activation='relu')))
    model.add((Conv2D(no_Of_Filters, size_Of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_Of_Pool))

    model.add((Conv2D(no_Of_Filters // 2, size_Of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_Of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_Of_Pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_Of_Classes, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical crossentropy', metrics=['accuracy'])
    return model
################################

# -----------------------------------------------------------

########## TRAIN ##########
model = myModel()
print(model.summary())

my_Callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

history = model.fit_generator(data_Gen.flow(X_train, y_train, batch_size=batch_Size_Val), steps_per_epoch=steps_Per_Epoch_Val, epochs=epochs_Val, validation_data=(X_validation))

################################

# -----------------------------------------------------------

########## PLOT ##########
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_Loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['acuracy'])
plt.plot(history.history['val_Acurracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')

plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print("\n Test Score: ", score[0])
print("\n Test Acurracy: ", score[1])
################################

# -----------------------------------------------------------

########## Store the model as a pickle object ##########
# pickle_Out = open("keras_model.h5", "wb")
# pickle.dump(model, pickle_Out)
# pickle_Out.close()
# cv2.waitKey(0)

model.save("my model")
model.save_weights("weights.h5")

################################





