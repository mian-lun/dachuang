import optuna
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import optuna.integration.lightgbm as oplgb
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt


file = "ETTh1.csv"
df = pd.read_csv(file)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(df['date'])
df.drop('date',axis=1,inplace=True)
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
# train_size = int(0.7*len(df))
train_target = len(df) - 720
train_size = int(0.7*train_target)
test_size = int(0.2*train_target)
val_size = train_target-train_size-test_size
# df = df.reset_index()


X = df.drop('OT',axis=1)
y = df['OT']
X_test_final = X.iloc[train_target:train_target+24,:]
y_test_final = y.iloc[train_target:train_target+24]


X = X.iloc[:train_target,:]
y = y.iloc[:train_target]


X_train, X_valid, X_test = X.iloc[:train_size,:], X.iloc[train_size:train_size+val_size,:], X.iloc[-test_size:,:]
y_train, y_valid, y_test = y.iloc[:train_size], y.iloc[train_size:train_size+val_size] ,y.iloc[-test_size:]

def objective(trial):
    param = {
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate',
                                                   [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        'n_estimators': trial.suggest_int('n_estimators',100, 800),
        'max_depth': trial.suggest_int('max_depth',6,70),

        # 'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, verbose=False)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse


study = optuna.create_study(direction='minimize',study_name='xgboost')
n_trials=50
study.optimize(objective, n_trials=n_trials)
print('Number of finished trials:', len(study.trials))
print("------------------------------------------------")
print('Best trial:', study.best_trial.params)
print("------------------------------------------------")
print(study.trials_dataframe())
print("------------------------------------------------")

params=study.best_params

# params = {'lambda': 0.0038575340019969527, 'alpha': 0.08009812283532823, 'colsample_bytree': 0.7, 'subsample': 1.0, 'learning_rate': 0.02, 'n_estimators': 303, 'max_depth': 36, 'min_child_weight': 35}
# params = {'lambda': 0.01470501281506617, 'alpha': 0.08343572112002091, 'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.018, 'n_estimators': 732, 'max_depth': 48, 'min_child_weight': 20}
# params = {'lambda': 0.0024544989825744444, 'alpha': 0.019930187638631507, 'colsample_bytree': 0.6, 'subsample': 0.8, 'learning_rate': 0.016, 'n_estimators': 603, 'max_depth': 32, 'min_child_weight': 279}
print(params)



model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, verbose=False)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test,preds)

print("MSE:",mse)
print("MAE:",mae)

fig = plt.figure(figsize=(16, 6))
plt.title(f'XGBoost MSE: {mse} and MAE: {mae}', fontsize=20)
plt.plot(y_test, color='red')
plt.plot(pd.Series(preds,index=y_test.index), color='green')
plt.xlabel('Time/H', fontsize=16)
plt.ylabel('WetBulbCelsius', fontsize=16)
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