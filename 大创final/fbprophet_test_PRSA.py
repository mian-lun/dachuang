import optuna
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score
import optuna.integration.lightgbm as oplgb
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

def mape_cal(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


tscv = TimeSeriesSplit(n_splits=5)

file = "PRSA2010-2014.csv"
df = pd.read_csv(file)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(df['date'])
df.drop('date',axis=1,inplace=True)

# train_size = int(0.7*len(df))
# train_target = len(df) - 720
train_size = int(0.7*len(df))
test_size = int(0.3*len(df))

df = df.reset_index()
df.rename(columns={'date': 'ds', 'pm2.5': 'y'}, inplace=True)

# train,valid, test = df.iloc[:train_size,:],df.iloc[train_size:,:], df.iloc[train_target:,:]
train, test = df.iloc[:train_size,:], df.iloc[-test_size:,:]
#
# def objective(trial):
#     param={
#         'n_changepoints': trial.suggest_int('n_changepoints',1,50),
#         'changepoint_range': trial.suggest_uniform('changepoint_range',0.05,0.95),
#         'seasonality_prior_scale': trial.suggest_uniform('seasonality_prior_scale',5,20),
#         'changepoint_prior_scale': trial.suggest_loguniform('changepoint_prior_scale',0.05,0.5),
#         'changepoint_range': trial.suggest_uniform('changepoint_range',0.8,1),
#         'seasonality_mode':trial.suggest_categorical('seasonality_mode',['multiplicative','additive']),
#         'yearly_seasonality':trial.suggest_categorical('yearly_seasonality',[True,False]),
#         'weekly_seasonality':trial.suggest_categorical('weekly_seasonality',[True,False]),
#         'daily_seasonality':trial.suggest_categorical('daily_seasonality',[True,False])
#
#     }
#     model = Prophet(**param)
#     model.add_regressor('DEWP')
#     model.add_regressor('TEMP')
#     model.add_regressor('PRES')
#     model.add_regressor('Iws')
#     model.add_regressor('Is')
#     model.add_regressor('Ir')
#
#
#     model.fit(train)
#     return cross_val_score(model,train,cv=tscv).mean()
#
# study = optuna.create_study(direction='minimize',study_name='fbprophet')
# n_trials=1
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
params = {'yearly_seasonality':True,'weekly_seasonality':True,'daily_seasonality':True}
model = Prophet(**params)
model.add_regressor('DEWP')
model.add_regressor('TEMP')
model.add_regressor('PRES')
model.add_regressor('Iws')
model.add_regressor('Is')
model.add_regressor('Ir')
model.fit(train)
preds = model.predict(test.drop('y', axis=1))
mse = mean_squared_error(test['y'], preds['yhat'])
mae = mean_absolute_error(test['y'], preds['yhat'])
mape = mape_cal(test['y'],preds['yhat'])

yhat_test = preds[['ds','yhat']]
yhat_test = yhat_test.set_index(yhat_test['ds'])
yhat_test.drop('ds',axis=1,inplace=True)
test = test[['ds','y']]
test = test.set_index(test['ds'])
test.drop('ds',axis=1,inplace=True)


print("MSE:",mse)
print("MAE:",mae)
print("MAPE:",mape)

fig = plt.figure(figsize=(16, 6))
plt.title(f'fbprophet MSE: {mse} and MAE: {mae} and MAPE: {mape}%', fontsize=20)
plt.plot(test['y'], color='red')
plt.plot(pd.Series(yhat_test['yhat']), color='green')
plt.xlabel('Hour', fontsize=16)
plt.ylabel('Number of Shared Bikes', fontsize=16)
plt.legend(labels=['Real', 'Prediction'], fontsize=16)
plt.grid()
plt.show()
