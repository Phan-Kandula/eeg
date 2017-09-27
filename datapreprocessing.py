import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from pandas import DataFrame

dataframe = pd.read_csv("data.csv")
inputX = dataframe.iloc[:, 1:].values
inputY = dataframe.iloc[:, 0]

print(type(inputY))
LabelEncoderY = LabelEncoder()
inputY = LabelEncoderY.fit_transform(inputY)
'''
ohc = OneHotEncoder(categorical_features=[0])
inputY = ohc.fit_transform(inputY)
'''
inputY = np.reshape(inputY, (inputY.size, 1))
print(type(inputY))
'''
inputY = DataFrame(inputY)
print(type(inputY))
'''
print(type(inputY))
print(inputY)
ohenc = OneHotEncoder(categorical_features="all")
inputY = ohenc.fit_transform(inputY).toarray()
print(inputY)