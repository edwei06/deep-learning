#car test linear regression model
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LinearRegression
dataframe_train = pd.read_csv(r'C:\Users\ed069\Desktop\diabetes\car_train.csv',header = None)
dataframe_test = pd.read_csv(r'C:\Users\ed069\Desktop\diabetes\car_test.csv',header = None)
datatest = dataframe_test.values
dataset = dataframe_train.values
X_train = dataset[:,0:5]
y_train = dataset[:,5]
X_test = datatest[:,0:5]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
poly_features = PolynomialFeatures(degree=(50))
X_train_poly = poly_features.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_train_poly,y_train)
model = make_pipeline(poly_features, poly_model)
predictions = model.predict(X_test)
print(predictions)