import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# from pandas import DataFrame
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

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

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


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
    arr = np.zeros(4)
    arr[np.argmax(pred)] = 1
    return arr

