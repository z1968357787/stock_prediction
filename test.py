import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pandas as pd
import sys

import config
import datetime
torch.manual_seed(0)
np.random.seed(0)

input_size=config.input_size
input_window = config.input_window
output_window = config.output_window
batch_size = config.batch_size
epochs = config.epochs
lr = config.lr
valid_loss=sys.float_info.max
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
dataset_name=config.dataset
# 创建时间序列
# 接下来需要对数据进行预处理，首先定义一个窗口划分的函数。
# 它的作用是将输入按照延迟output_windw的方式来划分数据以及其标签，
# 文中是进行单步预测，所以假设输入是1到20，则其标签就是2到21，
# 以适应Transformer的seq2seq的形式的输出。
def create_inout_sequences(input_data, tw):
    train_seqs = []
    train_labels=[]
    L = len(input_data)
    #print(input_data.shape)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw,:]
        #print(train_seq)
        train_label = input_data[i + 1,-1]
        #print(train_label.shape)
        #print(train_label)
        #train_seqs.append((train_seq, train_label))
        train_seqs.append(train_seq)
        train_labels.append(train_label)
    train_seqs = train_seqs[:-1]
    train_labels=train_labels[:-1]
    return torch.FloatTensor(train_seqs),torch.FloatTensor(train_labels)


def get_data(dataset):
    today = config.today
    #print(today)
    series = pd.read_csv('input/'+dataset)
    series.drop('ts_code', axis=1, inplace=True)  # 删除第二列’股票代码‘
    series = series.iloc[::-1]
    date = pd.to_datetime(series['trade_date'],format='%Y%m%d')
    # print(series)
    # print(series)
    # print(date)
    diff = date - today
    for i in range(len(diff)):
        diff[i] = diff[i].days
    # print(diff)
    series['trade_date'] = diff
    y = series['open']
    series.drop('open', axis=1, inplace=True)
    name = series.columns
    scaler = MinMaxScaler(feature_range=(-1, 1))
    series = scaler.fit_transform(series)
    series = pd.DataFrame(data=series, columns=name)
    # print(series)
    # print(y)
    series['open']=y
    #print(series)
    series=np.array(series)
    
    # train_samples = int(0.8 * len(series))
    # train_data = series[:train_samples]
    # test_data = series[train_samples:]
    test_data=series

    # train_sequence,train_labels = create_inout_sequences(train_data, input_window)

    test_data,test_labels = create_inout_sequences(test_data, input_window)

    return test_data.to(device),test_labels.to(device)


def get_batch(source,source_label, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    data_label=source_label[i:i+seq_len]
    return data, data_label

def evalue(data):
    prediction = data['prediction']
    y = data['open']
    return [mean_absolute_error(y,prediction),mean_squared_error(y,prediction),r2_score(y,prediction)]


def plot_and_loss(eval_model, data_source,data_label, epoch,model_name,dataset_name):
    global valid_loss
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source,data_label, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).cpu().item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    avg_loss=total_loss / i
    truth = truth.numpy()
    test_result = test_result.numpy()
    data = {'open': truth, 'prediction': test_result}
    result = pd.DataFrame(data=data)
    if dataset_name.endswith("-data.csv"):
        # result['prediction']=test_result.numpy()
        result.to_csv(f"output/predict_{model_name}_{dataset_name.split('-')[0]}.csv",index = False)

    return avg_loss,evalue(result)

model_name=config.model_name
model = torch.load(f'model/{model_name}-{dataset_name.split("-")[0]}.pth')
criterion = nn.MSELoss()
suffixs=["-data.csv","-train.csv","-test.csv"]
evals=[]
for suffix in suffixs:
    dataset_name2=dataset_name.split("-")[0]+suffix
    val_data, val_labels= get_data(dataset_name2)
    epoch_start_time = time.time()
    # train(train_data,train_labels)

    val_loss,eval_res = plot_and_loss(model, val_data, val_labels, 0, model_name, dataset_name2)
    evals.append(eval_res)

    # if (epoch % 10 is 0):
    #     val_loss = plot_and_loss(model, val_data,val_labels, epoch)
    # else:
    #     val_loss = evaluate(model, val_data,val_labels)

    print('-' * 89)
    print('| dataset name {:s} | time: {:5.2f}s | valid loss {:5.5f}'.format(dataset_name2, (
            time.time() - epoch_start_time), val_loss))
    # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
    #         time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)

eval=pd.DataFrame(evals,columns=['MAE','MSE','R2'],index = ['all','train','test'])
eval.to_csv(f"output/eval_{model_name}_{dataset_name.split('-')[0]}.csv")


