import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import scipy.stats as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import svm, datasets
from  sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y - 1) ** 2))
def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return 'rmspe', rmspe(y, yhat)

file = 'ETTh1.csv'
df = pd.read_csv(file)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(df['date'])
df.drop('date',axis=1,inplace=True)
train_size = len(df)-24

X = df.drop('OT',axis=1)
y = df['OT']

X_train, X_test = X.iloc[:train_size,:], X.iloc[train_size:,:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test,y_test)
watchlist = [(dtrain,'train'),(dtest,'eval')]
early_stop = 50
# # 模型初始化
# params = {"n_estimators": st.randint(100, 500),
#           'max_depth': [i for i in range(3, 10, 2)],
#           'min_child_weight': [i for i in range(1, 6, 2)],
#           'gamma':[i/10.0 for i in range(0,5)],
#           'subsample':[i/10.0 for i in range(6,10)],
#           'colsample_bytree':[i/10.0 for i in range(6,10)],
#           'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
#           }

params_sk = {
    'objective': 'reg:squarederror',
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'seed': 42}

skrg = xgb.XGBRegressor(**params_sk)
skrg.fit(X_train, y_train)

params_grid = {"n_estimators": st.randint(100, 500),
                  "colsample_bytree": st.beta(10, 1),
                  "subsample": st.beta(10, 1),
                  "gamma": st.uniform(0, 10),
                  'reg_alpha': st.expon(0, 50),
                  "min_child_weight": st.expon(0, 50),
                 "learning_rate": st.uniform(0.06, 0.12),
               'max_depth': st.randint(6, 30)
               }
search_sk = RandomizedSearchCV(
    skrg, params_grid, cv=5, random_state=1, n_iter=20)  # 5 fold cross validation
search_sk.fit(X, y)
print("best parameters:", search_sk.best_params_)
print(    "best score:", search_sk.best_score_)

params_new = {**params_sk, **search_sk.best_params_}

model_final = xgb.train(params_new, dtrain,200, evals=watchlist,
                        early_stopping_rounds=early_stop, verbose_eval=True)

yhat = model_final.predict(xgb.DMatrix(X_test))
mse = mean_squared_error(y_test,yhat)
mae = mean_absolute_error(y_test,yhat)
print("MSE:",mse)
print("MAE:",mae)




# num_boost_round = 6000
#
# dtrain = xgb.DMatrix(X_train,y_train)
# dtest = xgb.DMatrix(X_test,y_test)
# watchlist = [(dtrain,'train'),(dtest,'eval')]
#
# gsearch = GridSearchCV(
#     estimator = xgb.train(
#         learning_rate=0.1,
#     )
# )
#
#
# gbm = xgb.train(params,dtrain,num_boost_round,evals = watchlist,
#                 early_stopping_rounds = 100, feval = rmspe_xg, verbose_eval=True)
#
# yhat = gbm.predict(xgb.DMatrix(X_test))
#
#
#
#
# # xgb = xgb.XGBRegressor().fit(X_train,y_train)
# # predictions = xgb.predict(X_test)
# #
# mse = mean_squared_error(y_test,yhat)
# mae = mean_absolute_error(y_test,yhat)
#
# fig = plt.figure(figsize=(16, 6))
# plt.title(f'LightGBM MSE: {mse} and MAE: {mae}', fontsize=20)
# plt.plot(y_test, color='red')
# plt.plot(pd.Series(yhat, index=y_test.index), color='green')
# plt.xlabel('Hour', fontsize=16)
# plt.ylabel('Number of Shared Bikes', fontsize=16)
# plt.legend(labels=['Real', 'Prediction'], fontsize=16)
# plt.grid()
# plt.show()

