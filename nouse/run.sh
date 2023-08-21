echo "开始跑数据"
python3 get_data.py
python3 train_predict.py
echo "数据结束"
echo "开始画图"
python3 analyse.py
python3 draw_cmp.py
python3 draw_eval.py
python3 draw_time.py
python3 draw.py
echo "画图完毕"
echo "服务器启动"
cd Django/Web 
python3 manage.py runserver &
echo "服务器启动完毕"