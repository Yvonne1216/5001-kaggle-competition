import pandas as pd
import numpy as np
import xgboost as xgb
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

target = 'time'
IDcol = 'id'

train = pd.read_csv("/Users/yangfan/Desktop/Project/5001/individual/data/train.csv",index_col = 0)
test = pd.read_csv("/Users/yangfan/Desktop/Project/5001/individual/data/test.csv")
feat_col = [x for x in train.columns if x not in [target,IDcol]]

print(feat_col)
train=DataFrame(train)
test=DataFrame(test)

train[['penalty','alpha']]=train[['penalty','alpha']].apply(LabelEncoder().fit_transform)
test[['penalty','alpha']]=test[['penalty','alpha']].apply(LabelEncoder().fit_transform)

standardScaler = StandardScaler().fit(train[feat_col])
train[feat_col]= standardScaler.transform(train[feat_col])
standardScaler = StandardScaler().fit(test[feat_col])
test[feat_col]= standardScaler.transform(test[feat_col])
y_stdScaler = StandardScaler()
train_y = np.array(train[target])
train[target] = y_stdScaler.fit_transform(train_y.reshape(-1, 1))

dtrain = xgb.DMatrix(train[feat_col], train[target])
dtest = xgb.DMatrix(test[feat_col])

num_round = 300

params = {
    'booster': 'gbtree',
    # 'objective': 'binary:logistic',
    # 'subsample': 0.8,
    'colsample_bytree': 1,
    'eta': 0.3,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0.0,
    'silent': 1,
    # 'lambda':0.8,
    'eval_metric': 'error'
    }
bst = xgb.train(params, dtrain, num_round)
y = bst.predict(dtest)
y = y_stdScaler.inverse_transform(y)
pred_y = y.copy()
y = y.tolist()
y = [x if x > 0. else 0. for x in y ]

# Save data
file_name = "/Users/yangfan/Desktop/Project/5001/individual/data/result1.csv"
with open(file_name, "w") as des:
    des.write("Id,time\n")
    for idx, i in enumerate(y):
        des.write("%d,%.3f\n"%(idx, i))

