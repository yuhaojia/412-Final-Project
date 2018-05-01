import pandas as pd
import numpy as np
from sklearn import linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import multiclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, scale, MinMaxScaler, QuantileTransformer, \
    RobustScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
import pickle
import csv
from npy_to_csv import npy2csv


cat_str = 'Product_Info_1, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41'
con_str = 'Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5'
dis_str = 'Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32'

MLP = True
if MLP:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization, Activation, Softmax
    from keras.utils import np_utils
    from keras import optimizers
save_res = True


def prob_to_class(prob):
    output = list()
    for i in range(len(prob)):
        output.append(np.argmax(prob[i]) + 1)
    return output


def fill_missing(data):
    cat_list = cat_str.split(', ')
    con_list = con_str.split(', ')
    dis_list = dis_str.split(', ')
    col_name = list(data)
    dummy_list = []
    for col in col_name:
        if (col not in cat_list) and (col not in con_list) and (col not in dis_list):
            dummy_list.append(col)
    df = data.loc[:, cat_list]
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    df = pd.DataFrame(imp.fit_transform(df), columns=cat_list)
    df2 = data.loc[:, con_list]
    imp2 = Imputer(missing_values='NaN', strategy='mean', axis=0)
    df2 = pd.DataFrame(imp2.fit_transform(df2), columns=con_list)
    df3 = data.loc[:, dis_list]
    imp3 = Imputer(missing_values='NaN', strategy='median', axis=0)
    df3 = pd.DataFrame(imp3.fit_transform(df3), columns=dis_list)
    df4 = data.loc[:, dummy_list]
    imp4 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    df4 = pd.DataFrame(imp4.fit_transform(df4), columns=dummy_list)
    X = df.join(df2).join(df3).join(df4)
    return X


def preprocessing(data):
    data = data.drop(['Id'], 1)
    data = pd.get_dummies(data, columns=['Product_Info_2'], drop_first=True)
    X = fill_missing(data)

    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    # scaler = RobustScaler()

    X = scaler.fit_transform(X)
    # sel = SelectKBest(chi2, k=100)
    # X = sel.fit_transform(X, y)
    return X


def MLP_model():
    model = Sequential()
    model.add(Dense(512 * 2, input_dim=len(X[0]), kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(4096, kernel_initializer='uniform'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(512 * 2, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64 * 2, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model


def build_model():
    if MLP:
        model = MLP_model()
    else:
        # model = linear_model.LogisticRegression(C=1e5)
        # model = linear_model.LinearRegression()
        # model = multiclass.OneVsRestClassifier(estimator=svm.SVC(random_state=0, kernel='linear'))
        # model = svm.SVC(decision_function_shape='ovo')
        # model = DecisionTreeClassifier(random_state=0)
        model = RandomForestClassifier(max_depth=128, n_estimators=50, random_state=0)
        # model = MLPClassifier(solver="sgd", hidden_layer_sizes=(3,), activation="")
        # model = KNeighborsClassifier(n_neighbors=5)
        # model = AdaBoostClassifier(n_estimators=250, learning_rate=0.1)
    return model


def fit_model(model, X, y):
    if MLP:
        sgd = optimizers.SGD(lr=0.005, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, validation_split=0.3, epochs=1, batch_size=128, shuffle=True)
    else:
        model.fit(X, y)
        # Score
        print(cross_val_score(model, X, y, cv=5))
    return model


def predict(model, X):
    if MLP:
        res = model.predict_proba(X)
        res = prob_to_class(res)
    else:
        res = model.predict(X)
    if save_res:
        np.save('result', res)
        npy2csv('result.npy', "result" + str(np.random.randint(0, 1000)) + ".csv")


if __name__ == "__main__":
    data = pd.read_csv('training.csv')
    y = data.pop('Response')
    if MLP:
        # Encode label to one hot coding
        encoded_Y = LabelEncoder().fit(y).transform(y)
        y = np_utils.to_categorical(encoded_Y)
    X = preprocessing(data)
    model = build_model()
    model = fit_model(model, X, y)
    test_data = pd.read_csv('testing.csv')
    test_X = preprocessing(test_data)
    predict(model, test_X)
