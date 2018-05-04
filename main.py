import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import multiclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cross_validation import cross_val_score
from sklearn.utils import resample
from npy_to_csv import npy2csv
from LogisticRegression import LogisticRegression


categorical_str = 'Product_Info_1, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41'
continuous_str = 'Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5'
discrete_str = 'Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32'

Use_library = False
MLP = False
if MLP:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization, Activation
    from keras.utils import np_utils
save_res = True


def prob_to_class(prob):
    """
    Convert one-hot probability to class label
    :param prob: 2d-float array
    :return: 1d label array
    """
    output = list()
    for i in range(len(prob)):
        output.append(np.argmax(prob[i]) + 1)
    return output


def fill_missing(data):
    """
    Fill missing data with Imputer
    :param data:
    :return:
    """
    try:
        categorical_list = categorical_str.split(', ')
        new_categorical_list = []
        continuous_list = continuous_str.split(', ')
        discrete_list = discrete_str.split(', ')
        column_name = list(data)
        dummy_list = []
        for col in column_name:
            if (col not in categorical_list) and (col not in continuous_list) and (col not in discrete_list):
                dummy_list.append(col)
            if col in categorical_list:
                new_categorical_list.append(col)
        df = data.loc[:, categorical_list]
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        df = pd.DataFrame(imp.fit_transform(df), columns=categorical_list)
        df2 = data.loc[:, continuous_list]
        imp2 = Imputer(missing_values='NaN', strategy='mean', axis=0)
        df2 = pd.DataFrame(imp2.fit_transform(df2), columns=continuous_list)
        df3 = data.loc[:, discrete_list]
        imp3 = Imputer(missing_values='NaN', strategy='mean', axis=0)
        df3 = pd.DataFrame(imp3.fit_transform(df3), columns=discrete_list)
        df4 = data.loc[:, dummy_list]
        imp4 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        df4 = pd.DataFrame(imp4.fit_transform(df4), columns=dummy_list)
        X = df.join(df2).join(df3).join(df4)
    except:
        # If the features types are not provided, we applied mean imputer to all columns
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        X = imp.fit_transform(data)
    return X


def preprocessing(data, test_data):
    """
    Extract label from training data, fill missing data for both training and testing dataset, scaling
    :param data: training
    :param test_data: testing
    :return: X, testX, label
    """
    data = data.drop(['Id'], 1)
    test_data = test_data.drop(['Id'], 1)
    try:
        # print(data.Response.value_counts())
        for i in range(1, 8):
            if i == 4 or i == 3:
                df_minority = data[data.Response == i]
                df_sub = resample(df_minority, replace=True, n_samples=500, random_state=123)
                data = pd.concat([data, df_sub])
        # print(data.Response.value_counts())
        data = pd.get_dummies(data, columns=['Product_Info_2'], drop_first=True)
        test_data = pd.get_dummies(test_data, columns=['Product_Info_2'], drop_first=True)
    except:
        # it still can be applied to other dataset, as we will skip the resample and get dummy
        print('Not insurance dataset, need adjust')

    y = data.pop('Response')

    if MLP:
        # Encode label to one hot coding
        encoded_Y = LabelEncoder().fit(y).transform(y)
        y = np_utils.to_categorical(encoded_Y)

    X = fill_missing(data)
    test_X = fill_missing(test_data)

    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # scaler = RobustScaler()
    X = scaler.fit_transform(X)
    test_X = scaler.transform(test_X)
    return X, test_X, y


def MLP_model():
    """
    Define neural net model
    :return: model
    """
    model = Sequential()
    model.add(Dense(512 * 2, input_dim=len(X[0]), kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512 * 2, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512 // 2, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64 * 2, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model


def build_model():
    """
    Build model using keras, sklearn and self-define function
    :return: model
    """
    if MLP:
        model = MLP_model()
        model.summary()
    else:
        # model = LogisticRegression(C=1e5)
        model = LinearRegression()
        # model = multiclass.OneVsRestClassifier(estimator=svm.SVC(random_state=0, kernel='linear'))
        # model = svm.SVC(decision_function_shape='ovo')
        # model = DecisionTreeClassifier(random_state=0)
        # model = RandomForestClassifier(max_depth=64, n_estimators=50, random_state=123)
        # model = MLPClassifier(solver="sgd", hidden_layer_sizes=(3,), activation="")
        # model = KNeighborsClassifier(n_neighbors=5)
        # model = AdaBoostClassifier(n_estimators=250, learning_rate=0.1)
    return model


def fit_model(model, X, y):
    """
    Train model
    :param model:
    :param X: training data
    :param y: label
    :return: trained model
    """
    if MLP:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, validation_split=0.3, epochs=15, batch_size=256, shuffle=True)
    else:
        model.fit(X, y)
        print(cross_val_score(model, X, y, cv=5))
    return model


def predict(model, X):
    """
    Use trained model to predict risk for testing data
    :param model:
    :param X: testing data
    :return: prediction
    """
    if MLP:
        res = model.predict_proba(X)
        res = prob_to_class(res)
    else:
        res = model.predict(X)
    return res


if __name__ == "__main__":
    data = pd.read_csv('training.csv')
    test_data = pd.read_csv('testing.csv')
    X, test_X, y = preprocessing(data, test_data)

    # print(X.shape, test_X.shape)
    if Use_library:
        model = build_model()
        model = fit_model(model, X, y)
        res = predict(model, test_X)
    else:
        model = LogisticRegression(X, y, alpha=0.1, num_iters=50, regularized=True, normalization='l2')
        params = model.train(X, y, np.unique(y))
        classifedLabels = []
        for eachData in test_X:
            classifedLabels.append(model.classify(eachData, params))
        res = np.array(classifedLabels)
    if save_res:
        np.save('result', res)
        npy2csv('result.npy', "result" + str(np.random.randint(0, 1000)) + ".csv")