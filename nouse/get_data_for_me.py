import tushare as ts
import pandas as pd
import datetime
from sklearn import preprocessing

today = datetime.datetime(2001,1,1)
token = '35161dbcb4b76f532f815a1bee71c10a1b539c67a646462d9600616b'
pro = ts.pro_api(token)
ts_code='000615.sz'
data = pro.daily(ts_code=ts_code, start_date='20000101', end_date='20230818')
# date = pd.to_datetime(data['trade_date'])
# print(data)
# diff = date - today
# for i in range(len(diff)):
#     diff[i] = diff[i].days
# print(diff)
# data['trade_date'] = diff
# y = data['open']
# data = data.drop(["ts_code","open"],axis=1)
# print(data)
# name = data.columns
# data = preprocessing.StandardScaler().fit_transform(data)
# data = pd.DataFrame(data=data,columns=name)
# data['open'] = y
data.loc[0:0.8*len(data)].to_csv(f"./input/{ts_code.split('.')[0]}-train.csv",index=None)
data.loc[0.8*len(data):].to_csv(f"./input/{ts_code.split('.')[0]}-test.csv",index=None)
data.to_csv(f"./input/{ts_code.split('.')[0]}-data.csv",index=None)