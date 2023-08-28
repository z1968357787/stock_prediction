import os
import pandas as pd
# 要查看的目录路径
directory_path = 'output'
columns=['MLP','MLP2','MLP3','LSTM','GRU','RNN','TransAm']
index=['000628','000629','000635','000659','000663','000665','000666','000670','000679','000680']
mse=pd.DataFrame(columns=columns,index = index)
mae=pd.DataFrame(columns=columns,index = index)
r2=pd.DataFrame(columns=columns,index = index)
cal='test'
# 列出目录下的所有文件和子目录
file_list = os.listdir(directory_path)

# 遍历文件列表并输出文件名
for file_name in file_list:
    #print(file_name)
    prefix=file_name.split(".")
    strs=prefix[0].split("_")
    if strs[0]!='eval' or strs[2] == '000615':
        continue
    model=strs[1]
    dataset=strs[2]
    file=os.path.join(directory_path,file_name)
    res = pd.read_csv(file,index_col=0).round(4)
    mse.loc[dataset][model]=res.loc[cal]['MSE']
    mae.loc[dataset][model] = res.loc[cal]['MAE']
    r2.loc[dataset][model] = res.loc[cal]['R2']


mse.to_csv(f"analyse/MSE.csv")
mae.to_csv(f"analyse/MAE.csv")
r2.to_csv(f"analyse/R2.csv")
mse.to_excel(f"analyse/MSE.xlsx")
mae.to_excel(f"analyse/MAE.xlsx")
r2.to_excel(f"analyse/R2.xlsx")

