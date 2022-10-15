import optuna
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import optuna.integration.lightgbm as oplgb
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt


file = "WTH.csv"
df = pd.read_csv(file)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(df['date'])
df.drop('date',axis=1,inplace=True)

# train_size = int(0.7*len(df))
train_target = len(df) - 720
train_size = int(0.7*train_target)
test_size = int(0.2*train_target)
val_size = train_target-train_size-test_size


df = df.reset_index()
df.rename(columns={'date': 'ds', 'WetBulbCelsius': 'y'}, inplace=True)

# train,valid, test = df.iloc[:train_size,:],df.iloc[train_size:,:], df.iloc[train_target:,:]
train_tmp = df.iloc[:train_target,:]
train, valid, test = train_tmp.iloc[:train_size,:], train_tmp.iloc[train_size:train_size+val_size,:], train_tmp.iloc[-test_size:,:]

def objective(trial):
    param={
        # 'n_changepoints': trial.suggest_int('n_changepoints',1,50),
        # 'changepoint_range': trial.suggest_uniform('changepoint_range',0.05,0.95),
        'seasonality_prior_scale': trial.suggest_uniform('seasonality_prior_scale',5,20),
        'changepoint_prior_scale': trial.suggest_loguniform('changepoint_prior_scale',0.05,0.5),
        # 'changepoint_range': trial.suggest_uniform('changepoint_range',0.8,1),
        'seasonality_mode':trial.suggest_categorical('seasonality_mode',['multiplicative','additive']),
        # 'yearly_seasonality':trial.suggest_categorical('yearly_seasonality',[True,False]),
        # 'weekly_seasonality':trial.suggest_categorical('weekly_seasonality',[True,False]),
        # 'daily_seasonality':trial.suggest_categorical('daily_seasonality',[True,False])

    }
    model = Prophet(**param)
    model.add_regressor('Altimeter')
    model.add_regressor('DryBulbFarenheit')
    model.add_regressor('DryBulbCelsius')
    model.add_regressor('WetBulbFarenheit')
    model.add_regressor('DewPointFarenheit')
    model.add_regressor('DewPointCelsius')
    model.add_regressor('RelativeHumidity')
    model.add_regressor('WindSpeed')
    model.add_regressor('WindDirection')
    model.add_regressor('StationPressure')

    model.fit(train)
    preds = model.predict(valid.drop('y', axis=1))
    rmse = mean_squared_error(valid['y'], preds['yhat'], squared=False)
    return rmse

study = optuna.create_study(direction='minimize',study_name='fbprophet')
n_trials=100
study.optimize(objective, n_trials=n_trials)
print('Number of finished trials:', len(study.trials))
print("------------------------------------------------")
print('Best trial:', study.best_trial.params)
print("------------------------------------------------")
print(study.trials_dataframe())
print("------------------------------------------------")

params=study.best_params
print(params)

model = Prophet(**params)
model.fit(train)
preds = model.predict(test.drop('y', axis=1))
mse = mean_squared_error(test['y'], preds['yhat'])
mae = mean_absolute_error(test['y'], preds['yhat'])


yhat_test = preds[['ds','yhat']]
yhat_test = yhat_test.set_index(yhat_test['ds'])
yhat_test.drop('ds',axis=1,inplace=True)
test = test[['ds','y']]
test = test.set_index(test['ds'])
test.drop('ds',axis=1,inplace=True)


print("MSE:",mse)
print("MAE:",mae)


fig = plt.figure(figsize=(16, 6))
plt.title(f'fbprophet MSE: {mse} and MAE: {mae}', fontsize=20)
plt.plot(test['y'], color='red')
plt.plot(pd.Series(yhat_test['yhat']), color='green')
plt.xlabel('Hour', fontsize=16)
plt.ylabel('Number of Shared Bikes', fontsize=16)
plt.legend(labels=['Real', 'Prediction'], fontsize=16)
plt.grid()
plt.show()
