import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fbprophet import Prophet
from sklearn.metrics import mean_squared_error

# def mape(actual,pred):
#     actual,pred = np.array(actual), np.array(pred)
#     return np.mean(np.abs((actual-pred) / actual)) * 100



yuandan = pd.DataFrame({
  'holiday': 'yuandan',
  'ds': pd.to_datetime(['2017-01-01','2018-01-01']),
  'lower_window': 0,
  'upper_window': 1,
})
chunjie = pd.DataFrame({
  'holiday': 'chunjie',
  'ds': pd.to_datetime(['2017-01-28', '2018-02-16']),
  'lower_window': 0,
  'upper_window': 1,
})
guoqing = pd.DataFrame({
    'holiday':'guoqing',
    'ds' : pd.to_datetime(['2016-10-01','2017-10-01']),
    'lower_window':0,
    'upper_window':1,
})
zhongqiu = pd.DataFrame({
    'holiday':'zhongqiu',
    'ds' : pd.to_datetime(['2016-09-15','2017-10-04']),
    'lower_window':0,
    'upper_window':1,
})
laodong = pd.DataFrame({
    'holiday':'laodong',
    'ds' : pd.to_datetime(['2017-05-01','2018-05-01']),
    'lower_window':0,
    'upper_window':1,
})
duanwu = pd.DataFrame({
    'holiday':'duanwu',
    'ds' : pd.to_datetime(['2017-05-30','2018-06-18']),
    'lower_window':0,
    'upper_window':1,
})
holidays = pd.concat((yuandan, chunjie,guoqing,zhongqiu,laodong,duanwu))


file = "ETTh1.csv"
df = pd.read_csv(file)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(df['date'])
df.drop('date',axis=1,inplace=True)

# train_size = int(0.7*len(df))
train_size = len(df) - 720

df = df.reset_index()
df = df[['date','OT']]
df.rename(columns={'date': 'ds', 'OT': 'y'}, inplace=True)

train, test = df.iloc[:train_size,:], df.iloc[train_size:,:]

m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    n_changepoints=100,
    changepoint_range=0.95,
    seasonality_prior_scale=10.0,
    changepoint_prior_scale=0.05,
    holidays=holidays
)

m.fit(train)

predictions_train = m.predict(train.drop('y', axis=1))
predictions_test = m.predict(test.drop('y', axis=1))



mse = mean_squared_error(test['y'],predictions_test['yhat'])

yhat_test = predictions_test[['ds','yhat']]
yhat_test = yhat_test.set_index(yhat_test['ds'])
yhat_test.drop('ds',axis=1,inplace=True)

test = test.set_index(test['ds'])
test.drop('ds',axis=1,inplace=True)

# fig1 = m.plot_components(predictions_train)
# fig2 = m.plot(predictions_train)
# fig3 = m.plot_components(predictions_test)
# fig4 = m.plot(predictions_test)

fig = plt.figure(figsize=(16, 6))
plt.title(f'jieri Real vs Prediction - MSE {mse}', fontsize=20)
plt.plot(test['y'], color='red')
plt.plot(pd.Series(yhat_test['yhat']), color='green')
plt.xlabel('Hour', fontsize=16)
plt.ylabel('Number of Shared Bikes', fontsize=16)
plt.legend(labels=['Real', 'Prediction'], fontsize=16)
plt.grid()

plt.show()