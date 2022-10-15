from lightgbm import LGBMRegressor
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import optuna.integration.lightgbm as oplgb
import pandas as pd
from lightgbm import LGBMRegressor
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import optuna.integration.lightgbm as oplgb

file = "ETTh1.csv"
df = pd.read_csv(file)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(df['date'])
df.drop('date',axis=1,inplace=True)
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
# train_size = int(0.7*len(df))
train_size = len(df) - 24
# df = df.reset_index()


X = df.drop('OT',axis=1)
y = df['OT']

X_train, X_test = X.iloc[:train_size,:], X.iloc[train_size:,:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]


def objective(trial):
    param = {
        'metric': 'rmse',
        'random_state': 48,
        'n_estimators': 20000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20, 50]),
        'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth': trial.suggest_int('cat_smooth', 1, 100)
    }

    lgb = LGBMRegressor(**param)
    lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    pred_lgb = lgb.predict(X_test)
    rmse = mean_squared_error(y_test, pred_lgb, squared=False)
    return rmse

study=optuna.create_study(direction='minimize',study_name='LightGBM')
n_trials=200
study.optimize(objective, n_trials=n_trials)

params=study.best_params
print(params)
model = LGBMRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test,preds)

print("MSE:",mse)
print("MAE:",mae)