import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
print(train_df.head())
print(test_df.head())
label = train_df['Label']
del train_df['Label']
data_df = pd.concat((train_df, test_df), axis = 0)
print(data_df.head())
print(data_df.shape)
del data_df['Id']
sparse_feas = [col for col in data_df.columns if col[0] == 'C']
dense_feas = [col for col in data_df.columns if col[0] == 'I']
data_df[sparse_feas] = data_df[sparse_feas].fillna("-1")
data_df[dense_feas] = data_df[dense_feas].fillna(0)
for fea in sparse_feas:
    le = LabelEncoder()
    data_df[fea] = le.fit_transform(data_df[fea])
mms = MinMaxScaler()
data_df[dense_feas] = mms.fit_transform(data_df[dense_feas])
train = data_df[:train_df.shape[0]]
test = data_df[train_df.shape[0]:]
train['Label'] = label
train, val = train_test_split(train, test_size=0.2, random_state=2020)
train.reset_index(drop=True, inplace=True)
val.reset_index(drop=True, inplace=True)
print(train.head())
train.to_csv('data/processed_train.csv', index=0)
test.to_csv('data/processed_test.csv', index=0)
val.to_csv('data/processed_val.csv', index=0)
