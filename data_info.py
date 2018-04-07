import pandas as pd
import numpy as np



data = pd.read_csv('training.csv')

# print(data.shape)
#
# print(data.Product_Info_2.value_counts())
#
print(data.Response.value_counts())

# product_info = pd.get_dummies(data.Product_Info_2)
# data = pd.get_dummies(data, columns=['Product_Info_2'], drop_first=True)
#
# print(data.shape)

test = pd.read_csv('testing.csv')

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