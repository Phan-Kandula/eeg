import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from pandas import DataFrame

dataframe = pd.read_csv("data.csv")
inputX = dataframe.iloc[:, 1:].values
inputY = dataframe.iloc[:, 0]
LabelEncoderY = LabelEncoder()
inputY = LabelEncoderY.fit_transform(inputY)
inputY = np.reshape(inputY, (inputY.size, 1))
ohenc = OneHotEncoder(categorical_features="all")
inputY = ohenc.fit_transform(inputY).toarray()

