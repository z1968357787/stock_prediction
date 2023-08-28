# 数据获取以及预处理
python get_data.py # 修改文件中获取的数据集名称
# 数据集训练
python train.py # 修改config.py中对应的模型名和数据集名称
# 数据集预测
python test.py # 修改config.py中对应的模型名和数据集名称
# 结果绘图
python draw_cmp.py # 修改config2.py中对应的模型名和数据集名称
python draw_eval.py # 修改config2.py中对应的模型名和数据集名称
python draw_time.py # 修改config2.py中对应的模型名和数据集名称
python draw.py # 修改文件中数据集名称
# 系统运行
cd Django / Web
python manage.py runserver 127.0.0.1:8080 # &为自己的ip地址+端口号
