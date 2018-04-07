import pandas as pd
import numpy as np
from sklearn import linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import multiclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
import pickle
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import tensorflow as tf
from npy_to_csv import npy2csv


def prob_to_class(prob):
    output = list()
    for i in range(len(prob)):
        output.append(np.argmax(prob[i])+1)
    return output


data = pd.read_csv('training.csv')
data = data.drop(['Id'], 1)

y = data.pop('Response')

X = pd.get_dummies(data, columns=['Product_Info_2'], drop_first=True)
# X = data.drop(['Product_Info_2'], 1)
# Pre-process data
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)

# X = data.values[:, :-1]
X = StandardScaler().fit(X).transform(X)
print(X.shape, y.shape)



# Encode label to one hot coding
# encoded_Y = LabelEncoder().fit(y).transform(y)
# dummy_y = np_utils.to_categorical(encoded_Y)

# Built model
# model = linear_model.LogisticRegression(C=1e5)
# model = linear_model.LinearRegression()
# model = multiclass.OneVsRestClassifier(estimator=svm.SVC(random_state=0, kernel='linear'))
# model = DecisionTreeClassifier(random_state=0)
model = RandomForestClassifier(max_depth=8, n_estimators=10, random_state=0)
# model = MLPClassifier(solver="sgd", hidden_layer_sizes=(3,), activation="")
# model = KNeighborsClassifier(n_neighbors=5)

# model = Sequential()
# model.add(Dense(1024, activation='relu', input_dim=len(X[0])))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(8, activation='softmax'))

# Select parameters
# k_range = list(range(1, 11))
# param_grid = dict(n_neigbors=k_range)
# grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
# k_scores = []
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
#     k_scores.append(scores.mean())
# print(k_scores)


# Fit model
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.fit(X, dummy_y, validation_split=0.1, epochs=150, batch_size=64, shuffle=True)
model.fit(X, y)


# Save model
filename = 'random_forest_model.sav'
pickle.dump(model, open(filename, 'wb'))
# model.save('MLP_model.h5')
# Load model
# model = pickle.load(open(filename, 'rb'))

# Predict
test_data = pd.read_csv('testing.csv')
index = test_data.pop('Id')
test_data = pd.get_dummies(test_data, columns=['Product_Info_2'], drop_first=True)
# test_data = test_data.drop(['Product_Info_2'], 1)
test_data = imp.transform(test_data)
test_data = StandardScaler().fit(test_data).transform(test_data)
print(test_data.shape)

# res = model.predict_proba(test_data)
# res = prob_to_class(res)

res = model.predict(test_data)
print(res[0:5])
# res = np.argmax(res, axis=1)
np.save('result', res)
npy2csv('result.npy', "result"+str(np.random.randint(0, 1000))+".csv")
