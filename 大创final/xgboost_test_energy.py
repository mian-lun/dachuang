import optuna
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score, TimeSeriesSplit

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt


def mape_cal(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


tscv = TimeSeriesSplit(n_splits=5)

file = "energydata_complete.csv"
df = pd.read_csv(file)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(df['date'])
df.drop('date',axis=1,inplace=True)
df['hour'] = df.index.hour
df['day'] = df.index.day
# train_size = int(0.7*len(df))
# train_target = len(df) - 720
train_size = int(0.7*len(df))
test_size = int(0.3*len(df))
# df = df.reset_index()
# test_size = 250
# train_size = len(df)-test_size

X = df.drop('Appliances',axis=1)
y = df['Appliances']






X_train, X_test = X.iloc[:train_size,:], X.iloc[-test_size:,:]
y_train, y_test = y.iloc[:train_size] ,y.iloc[-test_size:]

# def objective(trial):
#     param = {
#         'lambda': trial.suggest_loguniform('lambda', 1e-1, 10.0),
#         # 'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
#         # 'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
#         # 'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
#         'learning_rate': trial.suggest_categorical('learning_rate',
#                                                    [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
#         'n_estimators': trial.suggest_int('n_estimators',100,300 ),
#         'max_depth': trial.suggest_int('max_depth',6,20),
#         #
#         # 'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
#         # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
#     }
#     model = xgb.XGBRegressor(**param)
#     model.fit(X_train, y_train)
#     # preds = model.predict(X_test)
#     # mape = mape_cal(y_test,preds)
#     return -cross_val_score(model,X_train,y_train,cv=tscv).mean()
#
#
# study = optuna.create_study(direction='minimize',study_name='xgboost')
# n_trials=100
# study.optimize(objective, n_trials=n_trials)
# print('Number of finished trials:', len(study.trials))
# print("------------------------------------------------")
# print('Best trial:', study.best_trial.params)
# print("------------------------------------------------")
# print(study.trials_dataframe())
# print("------------------------------------------------")
#
# params=study.best_params
# print(params)
params =  {'lambda': 0.35, 'learning_rate': 0.300000012, 'n_estimators': 100, 'max_depth': 6}
# params = {'lambda': 2.4525796787532386, 'learning_rate': 0.016, 'n_estimators': 104, 'max_depth': 6}
model = xgb.XGBRegressor(**params)
# model = xgb.XGBRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test,preds)
mape = mape_cal(y_test,preds)

print("MSE:",mse)
print("MAE:",mae)
print("MAPE:",mape)

fig = plt.figure(figsize=(16, 6))
plt.title(f'XGBoost MSE: {mse} and MAE: {mae} and MAPE: {mape}%' , fontsize=20)
plt.plot(y_test, color='red')
plt.plot(pd.Series(preds,index=y_test.index), color='green')
plt.xlabel('Time/H', fontsize=16)
plt.ylabel('PM2.5', fontsize=16)
plt.legend(labels=['Real', 'Prediction'], fontsize=16)
plt.grid()

plt.show()



#预测


# model = xgb.XGBRegressor(**params)
# model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, verbose=False)
# preds = model.predict(X_test_final)
# mse = mean_squared_error(y_test_final, preds)
# mae = mean_absolute_error(y_test_final,preds)
#
# print("MSE:",mse)
# print("MAE:",mae)
#
# fig = plt.figure(figsize=(16, 6))
# plt.title(f'XGBoost MSE: {mse} and MAE: {mae}', fontsize=20)
# plt.plot(y_test_final, color='red')
# plt.plot(pd.Series(preds,index=y_test_final.index), color='green')
# plt.xlabel('Time/H', fontsize=16)
# plt.ylabel('WetBulbCelsius', fontsize=16)
# plt.legend(labels=['Real', 'Prediction'], fontsize=16)
# plt.grid()
#
# plt.show()