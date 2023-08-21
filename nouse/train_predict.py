from pyspark import SparkContext, SparkConf
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import time
conf = SparkConf().setMaster('local').setAppName('MyApp')
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config("spark.driver.host","127.0.0.1")\
    .config("spark.ui.showConsoleProgress","false")\
    .appName("CruiseLinear").master("local[*]").getOrCreate()


import pandas as pd
import numpy as np
"""
读取数据
input: filename文件路径
output: spark sql的数据类型
"""
def read_datasets(name):
    data = spark.read.csv(name,inferSchema=True,header=True)
    vectorAssembler = VectorAssembler(inputCols=['trade_date', 'high', 'close', 'pre_close', 'change', 'pct_chg','vol','amount'],outputCol="features")
    data = vectorAssembler.transform(data)
    data = data.select(["features","open"])
    print(data)
    return data

"""
对于给定的预测结果，求MAE、MSE和R2
input: data 包含预测的数据和真实的数据
output: MAE、MSE和R2
"""
def evalue(data):
    prediction = data.toPandas()['prediction']
    y = data.toPandas()['open']
    return [mean_absolute_error(y,prediction),mean_squared_error(y,prediction),r2_score(y,prediction)]

"""
对于给定的模型和数据集，进行预测和评估
input: model 机器学习模型、train训练集、test测试集 data数据集
output: 一个3*3的表格，代表训练集、测试集和数据集的'MAE','MSE','R2'的值，运行时间和预测结果
"""
def predict_evalue(model,train,test,data):
    begin = time.time()
    model = model.fit(train)
    test_result = model.transform(test)
    train_result = model.transform(train)
    data_result = model.transform(data)
    end = time.time()
    return pd.DataFrame([evalue(test_result),evalue(train_result),evalue(data_result)],columns=['MAE','MSE','R2'],index = ['test','train','data_result']),end - begin,data_result.toPandas().drop('features',axis = 1)

"""
确定权重
input: 前面三个模型测试结果
output: 三个模型的权重
"""
def get_weight(m1,m2,m3):
    a = -0.25
    m1['MAE'],m2['MAE'],m3['MAE'] = a - m1['MAE'],a -  m2['MAE'],a -  m3['MAE']
    m1['MSE'],m2['MSE'],m3['MSE'] = a - m1['MSE'],a -  m2['MSE'],a -  m3['MSE']
    sum1, sum2, sum3 = m1.sum().sum(),m2.sum().sum(),m3.sum().sum()
    #print(sum1,sum2,sum3)
    sum_all = sum1 + sum2 + sum3
    #print([sum1 / sum_all,sum2 / sum_all,sum3 / sum_all])
    return [sum1 / sum_all,sum2 / sum_all,sum3 / sum_all]

"""
三个模型集成起来预测
input: 利用对应的权重进行预测
output: 预测结果 y
"""
def co_predict(weight,y1,y2,y3,y_true):
    y = []
    y1,y2,y3,y_true = list(y1), list(y2), list(y3),list(y_true)
    for i in range(len(y1)):
        y.append(weight[0] * y1[i] + weight[1] * y2[i] + weight[2] * y3[i])
    print(mean_absolute_error(y_true,y),mean_squared_error(y_true,y),r2_score(y_true,y))
    return y

if __name__ =="__main__":
    train = read_datasets("./input/train.csv")
    test = read_datasets("./input/test.csv")
    data = read_datasets("./input/data.csv")

    eval_1,time_1,predict_1 = predict_evalue(LinearRegression(labelCol="open"),train,test,data)
    eval_1.to_csv("./output/eval_LR.csv")
    predict_1.to_csv("./output/predict_LR.csv",index = False)
    eval_2,time_2,predict_2 = predict_evalue(GBTRegressor(labelCol="open"),train,test,data)
    eval_2.to_csv("./output/eval_GBT.csv")
    predict_2.to_csv("./output/predict_GBT.csv",index = False)
    eval_3,time_3,predict_3 = predict_evalue(RandomForestRegressor(labelCol="open"),train,test,data)
    eval_3.to_csv("./output/eval_RF.csv")
    predict_3.to_csv("./output/predict_RF.csv",index = False)
    print("用时",time_1,time_2,time_3)
    
    weight = get_weight(eval_1,eval_2,eval_3)
    y = co_predict(weight,predict_1['prediction'],predict_2['prediction'],predict_3['prediction'],predict_1['open'])
    pd.DataFrame(data=y,columns=['prediction']).to_csv("./output/my_predict.csv")
    