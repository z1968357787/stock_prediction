import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker
import pandas as pd
import datetime
import config2 as config
dataset=config.dataset
TransAm = pd.read_csv(config.predictions['TransAm'])
MLP = pd.read_csv(config.predictions['MLP'])
MLP2 = pd.read_csv(config.predictions['MLP2'])
MLP3 = pd.read_csv(config.predictions['MLP3'])
LSTM = pd.read_csv(config.predictions['LSTM'])
GRU = pd.read_csv(config.predictions['GRU'])
RNN = pd.read_csv(config.predictions['RNN'])
TransAm_val = list(TransAm['prediction'])
MLP_val = list(MLP['prediction'])
MLP2_val = list(MLP2['prediction'])
MLP3_val = list(MLP3['prediction'])
LSTM_val = list(LSTM['prediction'])
GRU_val = list(GRU['prediction'])
RNN_val = list(RNN['prediction'])
open_val = list(TransAm['open'])
#print(pred)
#Predict Result(Train: [0,400) Test: [400,464)])
begin = datetime.datetime(2000,1,1)
date = [datetime.timedelta(days=i) + begin for i in range(len(TransAm_val))]
date = [x.strftime('%Y.%m.%d') for x in date]
c = (
    Line(init_opts=opts.InitOpts(width="1140px", height="580px"))
    .add_xaxis(date)
    .add_yaxis("True_val", open_val, label_opts=opts.LabelOpts(is_show=False),symbol="triangle",color="green")
    .add_yaxis("MLP_Predict", MLP_val,label_opts=opts.LabelOpts(is_show=False),color = "red")
    .add_yaxis("MLP2_Predict", MLP2_val,label_opts=opts.LabelOpts(is_show=False),color = "blue")
    .add_yaxis("MLP3_Predict", MLP3_val,label_opts=opts.LabelOpts(is_show=False),color="orange")
    .add_yaxis("LSTM_Predict", LSTM_val,label_opts=opts.LabelOpts(is_show=False),color = "yellow")
    .add_yaxis("GRU_Predict", GRU_val,label_opts=opts.LabelOpts(is_show=False),color = "purple")
    .add_yaxis("RNN_Predict", RNN_val,label_opts=opts.LabelOpts(is_show=False),color="pink")
    .add_yaxis("TransAm_Predict", TransAm_val,label_opts=opts.LabelOpts(is_show=False),symbol = "cube",color="black")
    .set_global_opts(title_opts=opts.TitleOpts(title=""))
    .render(f"./Django/Web/user_manage/templates/cmp_{dataset}.html")
)
#print(Faker.choose())