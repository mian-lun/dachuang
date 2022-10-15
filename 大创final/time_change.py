import pandas as pd

#读取数据文件
df = pd.read_csv('AirQualityUCI.csv')

# #这里先把年月日合并转换格式为time，放入date列中
# df['date'] = df['year'].map(str)+"/"+df['month'].map(str)+"/"+df['day'].map(str)
# pd.to_datetime(df['date'])
#date列已经有年月日的数据，加上整数类型的小时数据转换为00:00:00的格式，将小时数据设为时间索引并合并
# df['date_time'] = pd.to_datetime(df['Date']) + pd.TimedeltaIndex(df['Time'],unit='H')
#删除不全的过渡列date
df['date_time'] = df['Date'] + ' ' + df['Time']
df=df.drop(['Date'],axis=1)

#打印查看效果
print(df['date_time'])
#将结果输出在另一个csv文件中
filename='testout_air.csv'
df.to_csv(filename)
