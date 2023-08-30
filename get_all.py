import tushare as ts
import pyecharts
# 设置Token
ts.set_token('35161dbcb4b76f532f815a1bee71c10a1b539c67a646462d9600616b') #请填入自己的Token
# 初始化接口
ts_api = ts.pro_api()
print("Tushare 接口调用正常，版本号:" + ts.__version__ +"\n")
print("pyecharts数据交互 接口调用正常，版本号:{}".format(pyecharts.__version__)+"\n")
stock_list = ts_api.stock_basic(exchange='', list_status='L',fields='ts_code,symbol,name,area,industry,market,list_date')
print(stock_list)
stock_list.to_csv("all/all-code.csv",index=None)