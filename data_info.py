import pandas as pd
import numpy as np
import sklearn.feature_selection
from sklearn.preprocessing import Imputer


pd.options.display.max_rows = 100
data = pd.read_csv('testing.csv')
data = data.drop(['Id'], 1)
data = pd.get_dummies(data, columns=['Product_Info_2'], drop_first=True)
print(data.nunique())
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

new_data = df.join(df2)

print(new_data.shape, df.shape, df2.shape)
print(new_data.columns)
# print(data.head())
# col_name = data.describe().columns
# print(col_name)
# i = col_name.get_loc('Medical_Keyword_1')
# print(i)
# data['Medical_Keyword_9'].fillna(20, inplace=True)
# # data[col_name.get_loc("Medical_Keyword_1"):col_name.get_loc("Medical_Keyword_48")].fillna(2, inplace=True)
# #
# print(data.iloc[1:20, col_name.get_loc("Medical_Keyword_1"):col_name.get_loc("Medical_Keyword_48")])
#
# print(data.select_dtypes(include=['float64']))
# describe = data.describe(include='all')
# print(data.groupby['Product_Info_1'].size())

# with open("describe.txt", 'w') as f:
#     for d in describe:
#         f.write(str(d))
# # print(data.shape)
# print(data.describe())
# print(data.Response.value_counts())
# missing = data.isnull().sum(axis=1).to_frame('nulls')
# print(missing.sort_values(by=['nulls']))

# row, col = data.shape
# drop_list = list()
# for i in range(col):
#     if data.iloc[:, i].isnull().sum() > row * 0.3:
#         drop_list.append(i)
# data.drop(data.columns[drop_list], axis=1, inplace=True)
#
# row, col = data.shape
# drop_list = list()
# for i in range(row):
#     if data.iloc[i, :].isnull().sum() > col * 0.1:
#         drop_list.append(i)
# data.drop(data.index[drop_list], axis=0, inplace=True)

# sel = sklearn.feature_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
# X = sel.fit_transform(data)
# print(X.shape)
# test = pd.read_csv('testing.csv')
# missing = test.isnull().sum().to_frame('nulls')
# print(missing.sort_values(by=['nulls']))
# print(test.shape)
#
# print(test.Product_Info_2.value_counts())

# test_product = pd.get_dummies(test.Product_Info_2)
# train_dummy, test_dummy = product_info.align(test_product, join='inner', axis=1)
# test = pd.get_dummies(test, columns=['Product_Info_2'], drop_first=True)
#
# print(test.shape)
#
# print(product_info.shape, test_product.shape, train_dummy.shape, test_dummy.shape)