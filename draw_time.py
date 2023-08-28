from pyecharts import options as opts
from pyecharts.charts import Grid, Liquid
from pyecharts.commons.utils import JsCode
import config2 as config
import pandas as pd
dataset=config.dataset
radiu=8
width="200px"
height="150px"
time = pd.read_csv(config.time_file)
time=time/60
column_means = time.mean()
#print(column_means)
row_means = column_means.sum()
#print(row_means)
print(time)
time=time/row_means
print(time)
#print(time_por)
l1 = Liquid(init_opts=opts.InitOpts(width=width, height=height)).add(
    "MLP",
    time['MLP'].tolist(),
    #outline_border_distance=radiu,
    center=["7.5%", "50%"],
    label_opts=opts.LabelOpts(
        font_size=25,
        formatter=JsCode(
            """function (param) {{
                    return (Math.floor(param.value * 1000) / 1000*{val}).toFixed(2) + ' M';
                }}""".format(val=row_means)
        ),
        position="inside",
    ),
)

l2 = Liquid(init_opts=opts.InitOpts(width=width, height=height)).add(
    "MLP2",
    time['MLP2'].tolist(),
    #outline_border_distance=radiu,
    center=["20%", "50%"],
    label_opts=opts.LabelOpts(
        font_size=25,
        formatter=JsCode(
            """function (param) {{
                    return (Math.floor(param.value * 1000) / 1000*{val}).toFixed(2) + ' M';
                }}""".format(val=row_means)
        ),
        position="inside",
    ),
)

l3 = Liquid(init_opts=opts.InitOpts(width=width, height=height)).add(
    "MLP3",
    time['MLP3'].tolist(),
    #outline_border_distance=radiu,
    center=["32.5%", "50%"],
    label_opts=opts.LabelOpts(
        font_size=25,
        formatter=JsCode(
            """function (param) {{
                    return (Math.floor(param.value * 1000) / 1000*{val}).toFixed(2) + ' M';
                }}""".format(val=row_means)
        ),
        position="inside",
    ),
)

l4 = Liquid(init_opts=opts.InitOpts(width=width, height=height)).add(
    "LSTM",
    time['LSTM'].tolist(),
    #outline_border_distance=radiu,
    center=["45%", "50%"],
    label_opts=opts.LabelOpts(
        font_size=25,
        formatter=JsCode(
            """function (param) {{
                    return (Math.floor(param.value * 1000) / 1000*{val}).toFixed(2) + ' M';
                }}""".format(val=row_means)
        ),
        position="inside",
    ),
)
l5 = Liquid(init_opts=opts.InitOpts(width=width, height=height)).add(
    "GRU",
    time['GRU'].tolist(),
    #outline_border_distance=radiu,
    center=["57.5%", "50%"],
    label_opts=opts.LabelOpts(
        font_size=25,
        formatter=JsCode(
            """function (param) {{
                    return (Math.floor(param.value * 1000) / 1000*{val}).toFixed(2) + ' M';
                }}""".format(val=row_means)
        ),
        position="inside",
    ),
)
l6 = Liquid(init_opts=opts.InitOpts(width=width, height=height)).add(
    "RNN",
    time['RNN'].tolist(),
    #outline_border_distance=radiu,
    center=["70%", "50%"],
    label_opts=opts.LabelOpts(
        font_size=25,
        formatter=JsCode(
            """function (param) {{
                    return (Math.floor(param.value * 1000) / 1000*{val}).toFixed(2) + ' M';
                }}""".format(val=row_means)
        ),
        position="inside",
    ),
)
l7 = Liquid(init_opts=opts.InitOpts(width=width, height=height)).add(
    "TransAm",
    time['TransAm'].tolist(),
    #outline_border_distance=radiu,
    center=["82.5%", "50%"],
    label_opts=opts.LabelOpts(
        font_size=25,
        formatter=JsCode(
            """function (param) {{
                    return (Math.floor(param.value * 1000) / 1000*{val}).toFixed(2) + ' M';
                }}""".format(val=row_means)
        ),
        position="inside",
    ),
)
grid = Grid(init_opts=opts.InitOpts(width="1300px", height="300px")).add(l1, grid_opts=opts.GridOpts()).add(l2, grid_opts=opts.GridOpts()).add(l3, grid_opts=opts.GridOpts()).add(l4, grid_opts=opts.GridOpts()).add(l5, grid_opts=opts.GridOpts()).add(l6, grid_opts=opts.GridOpts()).add(l7, grid_opts=opts.GridOpts())
grid.render(f"./Django/Web/user_manage/templates/time_{dataset}.html")
# grid = Grid(init_opts=opts.InitOpts(width="600px", height="400px")).add(l3, grid_opts=opts.GridOpts()).add(l4, grid_opts=opts.GridOpts())
# grid.render("./Django/Web/user_manage/time_2.html")
