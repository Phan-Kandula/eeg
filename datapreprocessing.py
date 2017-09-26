import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split

dataframe = pd.read_csv("data.csv")
inputX = dataframe.iloc[:, 1:].values
inputY = dataframe.iloc[:, 0].values

print(type(inputY))
inputY = np.reshape(inputY, (4,-1))
print(type(inputY))
LabelEncoderY = LabelEncoder()
inputY = LabelEncoderY.fit_transform(inputY)
ohc = OneHotEncoder(categorical_features=[0])
inputY = ohc.fit_transform(inputY)

print(inputY)
