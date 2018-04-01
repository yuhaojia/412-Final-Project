import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import Imputer
import pickle
import csv
from npy_to_csv import npy2csv


data = pd.read_csv('training.csv')
data = data.drop(['Id'], 1)
data = data.drop(['Product_Info_2'], 1)
X = data.values[:, :-1]
Y = data.values[:, -1]
print(X.shape, Y.shape)

# Pre-process data
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)

# Fit model
# model = linear_model.LogisticRegression(C=1e5)
# model.fit(X, Y)

# Save model
filename = 'logistic_model.sav'
# pickle.dump(model, open(filename, 'wb'))

# Load model
model = pickle.load(open(filename, 'rb'))

# Predict
test_data = pd.read_csv('testing.csv')
test_data = test_data.drop(['Id'], 1)
test_data = test_data.drop(['Product_Info_2'], 1)
test_data = imp.transform(test_data)
print(test_data.shape)

res = model.predict(test_data)
np.save('result', res)
npy2csv('result.npy', "result"+str(np.random.randint(0, 1000))+".csv")
