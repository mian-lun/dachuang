import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

from fbprophet import Prophet
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# def mape(actual,pred):
#     actual,pred = np.array(actual), np.array(pred)
#     return np.mean(np.abs((actual-pred) / actual)) * 100


def prophet_features(df,horizon=24*7):
    temp_df = df.reset_index()
    # temp_df = temp_df[['date', 'OT']]
    temp_df.rename(columns={'date':'ds','OT':'y'},inplace=True)
    train, test = temp_df.iloc[:horizon,:], temp_df.iloc[horizon:,:]


    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        n_changepoints=100,
        changepoint_range=0.95,
        seasonality_prior_scale=10.0,
        changepoint_prior_scale=0.05,
    )
    m.add_regressor('HUFL')
    m.add_regressor('HULL')
    m.add_regressor('MUFL')
    m.add_regressor('MULL')
    m.add_regressor('LUFL')
    m.add_regressor('LULL')

    m.fit(train)

    predictions_train = m.predict(train.drop('y',axis=1))
    predictions_test = m.predict(test.drop('y',axis=1))

    predictions = pd.concat([predictions_train,predictions_test],axis=0)

    return predictions


def lightGBM_prophet(df,horizon=24*7,lags=[1,2,3,4,5]):
    new_prophet_features = prophet_features(df,horizon=horizon)
    df.reset_index(inplace=True)
    df = pd.merge(df, new_prophet_features, left_on=['date'], right_on=['ds'], how='inner')
    df.drop('ds', axis=1, inplace=True)
    df.set_index('date', inplace=True)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month


    for lag in lags:
        df[f'yhat_lag_{lag}'] = df['yhat'].shift(lag)
    df.dropna(axis=0, how='any')

    X = df.drop('OT',axis=1)
    y = df['OT']

    X_train, X_test = X.iloc[:train_size, :], X.iloc[train_size:, :]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test,predictions)
    mae = mean_absolute_error(y_test,predictions)

    fig = plt.figure(figsize=(16, 6))
    plt.title(f'LightGBM MSE: {mse} and MAE: {mae}', fontsize=20)
    plt.plot(y_test, color='red')
    plt.plot(pd.Series(predictions, index=y_test.index), color='green')
    plt.xlabel('Hour', fontsize=16)
    plt.ylabel('Number of Shared Bikes', fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
    plt.show()

    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    plt.show()



file = "ETTh1.csv"
df = pd.read_csv(file)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(df['date'])
df.drop('date',axis=1,inplace=True)

# train_size = int(0.7*len(df))
train_size = len(df) - 24


warnings.simplefilter('ignore')
lightGBM_prophet(df,train_size)