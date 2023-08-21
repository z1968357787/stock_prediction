import tushare as ts
import pandas as pd
import datetime
from sklearn import preprocessing

today = datetime.datetime(2021,1,1)
token = '35161dbcb4b76f532f815a1bee71c10a1b539c67a646462d9600616b'
pro = ts.pro_api(token)
data = pro.daily(ts_code='000615.sz', start_date='20210101', end_date='20221130')
date = pd.to_datetime(data['trade_date'])
print(data)
diff = date - today
for i in range(len(diff)):
    diff[i] = diff[i].days
print(diff)
data['trade_date'] = diff
y = data['open']
data = data.drop(["ts_code","open"],axis=1)
print(data)
name = data.columns
data = preprocessing.StandardScaler().fit_transform(data)
data = pd.DataFrame(data=data,columns=name)
data['open'] = y
data.loc[0:400].to_csv("./input/train.csv",index=None)
data.loc[400:463].to_csv("./input/test.csv",index=None)
data.to_csv("./input/data.csv",index=None)