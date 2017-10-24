import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# from pandas import DataFrame
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
# from sklearn.metrics import confusion_matrix
from mover import Mover
import serial
import numpy as np
# data processing
data = pd.read_csv("data1.csv")
inputX = data.iloc[:, 1:].values
inputY = data.iloc[:, 0]
LabelEncoderY = LabelEncoder()
inputY = LabelEncoderY.fit_transform(inputY)
inputY = np.reshape(inputY, (inputY.size, 1))
ohenc = OneHotEncoder(categorical_features="all")
inputY = ohenc.fit_transform(inputY).toarray()

x_train, x_test, y_train, y_test = train_test_split(
    inputX, inputY, test_size=0.33, random_state=81)



def neural_model(x):
    classifier = Sequential()
    # Input layer
    # classifier.add(Dense(output_dim = len(x[0]), init = 'uniform', activation = 'relu',input_dim = len(x[0])))
    # hidden layer 1
    # classifier.add(Dense(output_dim = len(x[0]), init = 'uniform', activation =  'relu'))
    # hidden layer 2
    # classifier.add(Dense(output_dim = len(x[0]), init = 'uniform', activation = 'relu'))
    # hidden layer 3
    # classifier.add(Dense(output_dim = len(x[0]), init = 'uniform', activation = 'relu'))
    # outpute layer
    # classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'softmax'))
    # print(classifier)
    # return classifier

    classifier.add(Dense(units=len(
        x[0]), kernel_initializer='uniform', activation='relu', input_dim=len(x[0])))
    classifier.add(Dense(kernel_initializer='uniform',
                         activation='relu', units=len(x[0])))
    classifier.add(Dense(kernel_initializer='uniform',
                         activation='relu', units=len(x[0])))
    classifier.add(Dense(kernel_initializer="uniform",
                         units=4, activation="softmax"))

    return classifier


ann = neural_model(x_train)
ann.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=10, epochs=10)

def prediction(x):
    pred = ann.predict(x)
    direction = {0 : "down", 1 : "left", 2 : "right", 3 : "up"}
    return direction[np.argmax(pred)]

mover = Mover()
ser = serial.Serial("COM8", 9600, timeout=None)


def line_to_nparray(line):
    arr = line.split(',')
    arr = list(map(int, arr))
    return np.array(arr, dtype=np.int64, ndmin=2)


def move_mouse(line):
    side = prediction(line)
    if side == 'down':
        mover.move_down()
    elif side == 'left':
        mover.move_left()
    elif side == 'right':
        mover.move_right()
    elif side == 'up':
        mover.move_up

line = ''
while (True):
    c = ser.read()
    if c == b'\t':
        while(True):
            c = ser.read()
            if (c != b'\n' and c != b'\r'):
                line = line + c.decode("utf-8")
            elif(c == b'\r'):
                line = line_to_nparray(line)
                move_mouse(line)
                line = ''
                break
            else:
                line = line_to_nparray(line)
                move_mouse(line)
                line = ''
                break


