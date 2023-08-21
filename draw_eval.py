from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.charts import Bar, Timeline
from pyecharts.faker import Faker
import config2 as config
import pandas as pd
dataset=config.dataset
TransAm = pd.read_csv(config.evals['TransAm'],index_col=0).round(2)
MLP = pd.read_csv(config.evals['MLP'],index_col=0).round(2)
MLP2 = pd.read_csv(config.evals['MLP2'],index_col=0).round(2)
MLP3 = pd.read_csv(config.evals['MLP3'],index_col=0).round(2)
LSTM = pd.read_csv(config.evals['LSTM'],index_col=0).round(2)
GRU = pd.read_csv(config.evals['GRU'],index_col=0).round(2)
RNN = pd.read_csv(config.evals['RNN'],index_col=0).round(2)
MSE='MSE'
MAE='MAE'
R2='R2'
train='train'
test='test'
all='all'
x = Faker.choose()
tl = Timeline(init_opts=opts.InitOpts(width="1140px", height="500px"))
c = (
    Bar(init_opts=opts.InitOpts(width="1140px", height="580px"))
    .add_xaxis(
        [
            "MLP",
            "MLP2",
            "MLP3",
            "LSTM",
            "GRU",
            "RNN",
            "TransAm",
        ]
    )
    .add_yaxis("训练集", [MLP.loc[train,MSE], MLP2.loc[train,MSE], MLP3.loc[train,MSE],LSTM.loc[train,MSE],GRU.loc[train,MSE],RNN.loc[train,MSE],TransAm.loc[train,MSE]])
    .add_yaxis("测试集",  [MLP.loc[test,MSE], MLP2.loc[test,MSE], MLP3.loc[test,MSE],LSTM.loc[test,MSE],GRU.loc[test,MSE],RNN.loc[test,MSE],TransAm.loc[test,MSE]])
    .add_yaxis("所有数据", [MLP.loc[all,MSE], MLP2.loc[all,MSE], MLP3.loc[all,MSE],LSTM.loc[all,MSE],GRU.loc[all,MSE],RNN.loc[all,MSE],TransAm.loc[all,MSE]])
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
        title_opts=opts.TitleOpts(title="", subtitle=""),
    )
    #.render("./Django2/Web/user_manage/eval_MSE.html")
)
tl.add(c, "MSE")
c = (
    Bar(init_opts=opts.InitOpts(width="1140px", height="580px"))
    .add_xaxis(
        [
            "MLP",
            "MLP2",
            "MLP3",
            "LSTM",
            "GRU",
            "RNN",
            "TransAm",
        ]
    )
    .add_yaxis("训练集",
               [MLP.loc[train, MAE], MLP2.loc[train, MAE], MLP3.loc[train, MAE], LSTM.loc[train, MAE], GRU.loc[train, MAE], RNN.loc[train, MAE],
                TransAm.loc[train, MAE]])
    .add_yaxis("测试集",
               [MLP.loc[test, MAE], MLP2.loc[test, MAE], MLP3.loc[test, MAE], LSTM.loc[test, MAE], GRU.loc[test, MAE], RNN.loc[test, MAE],
                TransAm.loc[test, MAE]])
    .add_yaxis("所有数据", [MLP.loc[all, MAE], MLP2.loc[all, MAE], MLP3.loc[all, MAE], LSTM.loc[all, MAE], GRU.loc[all, MAE], RNN.loc[all, MAE],
                            TransAm.loc[all, MAE]])
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
        title_opts=opts.TitleOpts(title="", subtitle=""),
    )
    #.render("./Django2/Web/user_manage/eval_MAE.html")
)
tl.add(c, "MAE")
c = (
    Bar(init_opts=opts.InitOpts(width="1140px", height="580px"))
    .add_xaxis(
        [
            "MLP",
            "MLP2",
            "MLP3",
            "LSTM",
            "GRU",
            "RNN",
            "TransAm",
        ]
    )
    .add_yaxis("训练集",
               [MLP.loc[train, R2], MLP2.loc[train, R2], MLP3.loc[train, R2], LSTM.loc[train, R2], GRU.loc[train, R2], RNN.loc[train, R2],
                TransAm.loc[train, R2]])
    .add_yaxis("测试集",
               [MLP.loc[test, R2], MLP2.loc[test, R2], MLP3.loc[test, R2], LSTM.loc[test, R2], GRU.loc[test, R2], RNN.loc[test, R2],
                TransAm.loc[test, R2]])
    .add_yaxis("所有数据", [MLP.loc[all, R2], MLP2.loc[all, R2], MLP3.loc[all, R2], LSTM.loc[all, R2], GRU.loc[all, R2], RNN.loc[all, R2],
                            TransAm.loc[all, R2]])
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
        title_opts=opts.TitleOpts(title="", subtitle=""),
    )
    #.render("./Django2/Web/user_manage/eval_R2.html")
)
tl.add(c, "R2")


    #print(type(bar))
tl.render(f"./Django/Web/user_manage/generator/eval_all_{dataset}.html")