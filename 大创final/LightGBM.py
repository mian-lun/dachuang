import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

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

model = LGBMRegressor(num_leaves=100,learning_rate=0.1,random_state=42)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
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

