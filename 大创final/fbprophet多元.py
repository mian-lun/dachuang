import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# def mape(actual,pred):
#     actual,pred = np.array(actual), np.array(pred)
#     return np.mean(np.abs((actual-pred) / actual)) * 100



file = "ETTh1.csv"
df = pd.read_csv(file)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(df['date'])
df.drop('date',axis=1,inplace=True)

# train_size = int(0.7*len(df))
train_size = len(df) - 24

df = df.reset_index()
df.rename(columns={'date': 'ds', 'OT': 'y'}, inplace=True)

train, test = df.iloc[:train_size,:], df.iloc[train_size:,:]
param = {'n_changepoints': 50, 'changepoint_range': 0.7879359015051732, 'seasonality_prior_scale': 10.885997167111107, 'changepoint_prior_scale': 0.057660533142963655, 'seasonality_mode': 'multiplicative', 'yearly_seasonality': True, 'weekly_seasonality': False, 'daily_seasonality': True}

m = Prophet(
    **param
)
m.add_regressor('HUFL')
m.add_regressor('HULL')
m.add_regressor('MUFL')
m.add_regressor('MULL')
m.add_regressor('LUFL')
m.add_regressor('LULL')


m.fit(train.iloc[:int(0.7*train_size),:])

predictions_train = m.predict(train.drop('y', axis=1))
predictions_test = m.predict(test.drop('y', axis=1))
mse = mean_squared_error(test['y'],predictions_test['yhat'])
mae = mean_absolute_error(test['y'],predictions_test['yhat'])


yhat_test = predictions_test[['ds','yhat']]
yhat_test = yhat_test.set_index(yhat_test['ds'])
yhat_test.drop('ds',axis=1,inplace=True)
test = test[['ds','y']]
test = test.set_index(test['ds'])
test.drop('ds',axis=1,inplace=True)

fig1 = m.plot_components(predictions_train)
fig2 = m.plot(predictions_train)
fig3 = m.plot_components(predictions_test)
fig4 = m.plot(predictions_test)

fig = plt.figure(figsize=(16, 6))
plt.title(f'fbprophet MSE: {mse} and MAE: {mae}', fontsize=20)
plt.plot(test['y'], color='red')
plt.plot(pd.Series(yhat_test['yhat']), color='green')
plt.xlabel('Hour', fontsize=16)
plt.ylabel('Number of Shared Bikes', fontsize=16)
plt.legend(labels=['Real', 'Prediction'], fontsize=16)
plt.grid()

plt.show()