from models import TransAm,MLP,MLP2,MLP3,LSTM,GRU,RNN
import datetime
import torch
dataset_select=0
datasets=['000615-data.csv','000628-data.csv','000629-data.csv','000635-data.csv','000659-data.csv','000663-data.csv','000665-data.csv','000666-data.csv','000670-data.csv','000679-data.csv','000680-data.csv']
dataset=datasets[dataset_select]
model_select=0
model_names=['TransAm','MLP','MLP2','MLP3','LSTM','GRU','RNN']
model_name= model_names[model_select]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size=10
input_window = 7
output_window = 1
batch_size = 64
epochs = 200
lr = 0.005
feature_size=250
num_layers=1
dropout=0.0
today = datetime.datetime(2000, 1, 1)

models={'TransAm':TransAm(input_size,input_window=input_window),
        'MLP':MLP(input_size,input_window=input_window),
        'MLP2':MLP2(input_size,hidden_size=feature_size,input_window=input_window),
        'MLP3':MLP3(input_size,hidden_size=feature_size,input_window=input_window),
        'LSTM':LSTM(input_size,hidden_size=feature_size,device=device),
        'GRU':GRU(input_size,hidden_size=feature_size,device=device),
        'RNN':RNN(input_size,hidden_size=feature_size,device=device)}

model=models[model_name]