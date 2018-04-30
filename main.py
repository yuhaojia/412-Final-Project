import pandas as pd
import numpy as np
from sklearn import linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import multiclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, scale, MinMaxScaler, QuantileTransformer, RobustScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
import pickle
import csv
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.utils import np_utils
# import tensorflow as tf
from npy_to_csv import npy2csv


def prob_to_class(prob):
    output = list()
    for i in range(len(prob)):
        output.append(np.argmax(prob[i])+1)
    return output


data = pd.read_csv('training.csv')
data = data.drop(['Id'], 1)
y = data.pop('Response')
# X = data.drop(['Product_Info_2'], 1)
# Pre-process data

# data['Product_Info_1'].fillna(1, inplace=True)
# data['Product_Info_3'].fillna(26, inplace=True)
# data['Product_Info_5'].fillna(2, inplace=True)
# data['Product_Info_6'].fillna(3, inplace=True)
# data['Product_Info_7'].fillna(1, inplace=True)
# data['Medical_Keyword_1'].fillna(0, inplace=True)
# data['Medical_Keyword_2'].fillna(0, inplace=True)
# data['Medical_Keyword_3'].fillna(0, inplace=True)
# data['Medical_Keyword_4'].fillna(0, inplace=True)
# data['Medical_Keyword_5'].fillna(0, inplace=True)
# data['Medical_Keyword_6'].fillna(0, inplace=True)
# data['Medical_Keyword_7'].fillna(0, inplace=True)
# data['Medical_Keyword_8'].fillna(0, inplace=True)
# data['Medical_Keyword_9'].fillna(0, inplace=True)
# data['Medical_Keyword_10'].fillna(0, inplace=True)
# data['Medical_Keyword_11'].fillna(0, inplace=True)
# data['Medical_Keyword_12'].fillna(0, inplace=True)
# data['Medical_Keyword_13'].fillna(0, inplace=True)
# data['Medical_Keyword_14'].fillna(0, inplace=True)
# data['Medical_Keyword_15'].fillna(0, inplace=True)
# data['Medical_Keyword_16'].fillna(0, inplace=True)
# data['Medical_Keyword_17'].fillna(0, inplace=True)
# data['Medical_Keyword_18'].fillna(0, inplace=True)
# data['Medical_Keyword_19'].fillna(0, inplace=True)
# data['Medical_Keyword_20'].fillna(0, inplace=True)
# data['Medical_Keyword_21'].fillna(0, inplace=True)
# data['Medical_Keyword_22'].fillna(0, inplace=True)
# data['Medical_Keyword_23'].fillna(0, inplace=True)
# data['Medical_Keyword_24'].fillna(0, inplace=True)
# data['Medical_Keyword_25'].fillna(0, inplace=True)
# data['Medical_Keyword_26'].fillna(0, inplace=True)
# data['Medical_Keyword_27'].fillna(0, inplace=True)
# data['Medical_Keyword_28'].fillna(0, inplace=True)
# data['Medical_Keyword_29'].fillna(0, inplace=True)
# row, col = data.shape
# drop_list = list()
# for i in range(col):
#     if data.iloc[:, i].isnull().sum() > row * 0.5:
#         drop_list.append(i)
# data.drop(data.columns[drop_list], axis=1, inplace=True)

# row, col = data.shape
# drop_list_tmp = list()
# for i in range(row):
#     if data.iloc[i, :].isnull().sum() > col * 0.1:
#         drop_list_tmp.append(i)
# data.drop(data.index[drop_list_tmp], axis=0, inplace=True)

data = pd.get_dummies(data, columns=['Product_Info_2'], drop_first=True)

count = data.nunique()
cat_list = list()
num_list = list()
for i in range(len(count)):
    if count[i] > 3:
        num_list.append(i)
    else:
        cat_list.append(i)
print(num_list)
print(cat_list)
col_name = list(data)
num_col = []
cat_col = []
for elem in num_list:
    num_col.append(col_name[elem])
for elem in cat_list:
    cat_col.append(col_name[elem])

df = data.loc[:, num_col]
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df = pd.DataFrame(imp.fit_transform(df), columns=num_col)
df2 = data.loc[:, cat_col]
imp2 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
df2 = pd.DataFrame(imp2.fit_transform(df2), columns=cat_col)

X = df.join(df2)

# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

# scaler = StandardScaler()
scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer(random_state=0)
X = scaler.fit_transform(X)
# sel = SelectKBest(chi2, k=100)
# X = sel.fit_transform(X, y)
print(X.shape, y.shape)



# Encode label to one hot coding
# encoded_Y = LabelEncoder().fit(y).transform(y)
# dummy_y = np_utils.to_categorical(encoded_Y)

# Built model
model = linear_model.LogisticRegression(C=1e5)
# model = linear_model.LinearRegression()
# model = multiclass.OneVsRestClassifier(estimator=svm.SVC(random_state=0, kernel='linear'))
# model = DecisionTreeClassifier(random_state=0)
# model = RandomForestClassifier(max_depth=8, n_estimators=10, random_state=0)
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
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, dummy_y, validation_split=0.1, epochs=50, batch_size=64, shuffle=True)
model.fit(X, y)


# Score
print(cross_val_score(model, X, y, cv=5))

# Save model
# filename = 'random_forest_model.sav'
# pickle.dump(model, open(filename, 'wb'))
# model.save('MLP_model.h5')
# Load model
# model = pickle.load(open(filename, 'rb'))

# Predict
test_data = pd.read_csv('testing.csv')
index = test_data.pop('Id')
# test_data['Product_Info_1'].fillna(1, inplace=True)
# # test_data['Product_Info_3'].fillna(26, inplace=True)
# test_data['Product_Info_5'].fillna(2, inplace=True)
# test_data['Product_Info_6'].fillna(3, inplace=True)
# test_data['Product_Info_7'].fillna(1, inplace=True)
# test_data['Medical_Keyword_1'].fillna(0, inplace=True)
# test_data['Medical_Keyword_2'].fillna(0, inplace=True)
# test_data['Medical_Keyword_3'].fillna(0, inplace=True)
# test_data['Medical_Keyword_4'].fillna(0, inplace=True)
# test_data['Medical_Keyword_5'].fillna(0, inplace=True)
# test_data['Medical_Keyword_6'].fillna(0, inplace=True)
# test_data['Medical_Keyword_7'].fillna(0, inplace=True)
# test_data['Medical_Keyword_8'].fillna(0, inplace=True)
# test_data['Medical_Keyword_9'].fillna(0, inplace=True)
# test_data['Medical_Keyword_10'].fillna(0, inplace=True)
# test_data['Medical_Keyword_11'].fillna(0, inplace=True)
# test_data['Medical_Keyword_12'].fillna(0, inplace=True)
# test_data['Medical_Keyword_13'].fillna(0, inplace=True)
# test_data['Medical_Keyword_14'].fillna(0, inplace=True)
# test_data['Medical_Keyword_15'].fillna(0, inplace=True)
# test_data['Medical_Keyword_16'].fillna(0, inplace=True)
# test_data['Medical_Keyword_17'].fillna(0, inplace=True)
# test_data['Medical_Keyword_18'].fillna(0, inplace=True)
# test_data['Medical_Keyword_19'].fillna(0, inplace=True)
# test_data['Medical_Keyword_20'].fillna(0, inplace=True)
# test_data['Medical_Keyword_21'].fillna(0, inplace=True)
# test_data['Medical_Keyword_22'].fillna(0, inplace=True)
# test_data['Medical_Keyword_23'].fillna(0, inplace=True)
# test_data['Medical_Keyword_24'].fillna(0, inplace=True)
# test_data['Medical_Keyword_25'].fillna(0, inplace=True)
# test_data['Medical_Keyword_26'].fillna(0, inplace=True)
# test_data['Medical_Keyword_27'].fillna(0, inplace=True)
# test_data['Medical_Keyword_28'].fillna(0, inplace=True)
# test_data['Medical_Keyword_29'].fillna(0, inplace=True)
# test_data.drop(test_data.columns[drop_list], axis=1, inplace=True)
test_data = pd.get_dummies(test_data, columns=['Product_Info_2'], drop_first=True)
# test_data = test_data.drop(['Product_Info_2'], 1)

df = test_data.loc[:, num_col]
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df = pd.DataFrame(imp.fit_transform(df), columns=num_col)
df2 = test_data.loc[:, cat_col]
imp2 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
df2 = pd.DataFrame(imp2.fit_transform(df2), columns=cat_col)

test_data = df.join(df2)

# test_data = imp.transform(test_data)
test_data = scaler.transform(test_data)
# test_data = sel.transform(test_data)
print(test_data.shape)

# res = model.predict_proba(test_data)
# res = prob_to_class(res)

res = model.predict(test_data)

# np.save('result', res)
# npy2csv('result.npy', "result"+str(np.random.randint(0, 1000))+".csv")
