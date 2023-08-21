import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
import os

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
    #series.drop('id', axis=1, inplace=True)  # 删除第一列’id‘
    #series.drop('pre_close', axis=1, inplace=True)  # 删除列’pre_close‘
    #series.drop('trade_date', axis=1, inplace=True)  # 删除列’trade_date‘
    series=series.iloc[::-1]
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
    
    train_samples = int(0.8 * len(series))
    train_data = series[:train_samples]
    test_data = series[train_samples:]

    train_sequence,train_labels = create_inout_sequences(train_data, input_window)

    test_data,test_labels = create_inout_sequences(test_data, input_window)

    return train_sequence.to(device),train_labels.to(device), test_data.to(device),test_labels.to(device)

#data1,data2=get_data()
#print(data1[0])

def get_batch(source,source_label, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    data_label=source_label[i:i+seq_len]
    # print(data.shape)
    # input = torch.stack(torch.stack([item for item in data]).chunk(input_window, 1))
    # input=input.view(input.size(0),input.size(1),-1)
    # print(data.size())
    # target = torch.stack(torch.stack([item for item in data_label]).chunk(input_window, 1))
    # print(data.shape)
    # print(data_label.shape)
    return data, data_label


def train(train_data,train_labels):
    model.train()

    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        start_time = time.time()
        total_loss = 0
        data, targets = get_batch(train_data,train_labels, i, batch_size)
        # print(targets.shape)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f}'
                  .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_lr()[0],
                          elapsed * 1000 / log_interval, cur_loss))
            # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'
            #       .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_lr()[0],
            #               elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))

def evaluate(eval_model, data_source,data_labels):
    eval_model.eval()
    total_loss = 0
    eval_batch_size = 1
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source,data_labels, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)


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
    #print(valid_loss)
    if avg_loss < valid_loss:
        valid_loss = avg_loss
        plt.plot(test_result, color="red")
        plt.plot(truth, color="blue")
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.savefig(f'graph/{model_name}-{dataset_name.split("-")[0]}.png')
        plt.close()
        #print(model_name)
        torch.save(model, f'model/{model_name}-{dataset_name.split("-")[0]}.pth')
        print('-' * 89)
        print("save model")
    return avg_loss


train_data,train_labels, val_data,val_labels = get_data(dataset_name)
#model = MLP(10,input_window=7).to(device)
model_name=config.model_name
model = config.model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

start_time = time.time()
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data,train_labels)

    val_loss = plot_and_loss(model, val_data, val_labels, epoch, model_name, dataset_name)

    # if (epoch % 10 is 0):
    #     val_loss = plot_and_loss(model, val_data,val_labels, epoch)
    # else:
    #     val_loss = evaluate(model, val_data,val_labels)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f}'.format(epoch, (
            time.time() - epoch_start_time), val_loss))
    # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
    #         time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)
    scheduler.step()

end_time=time.time()
train_time=[end_time-start_time]
time_file_path=f'output/time_{dataset_name.split("-")[0]}.csv'
if os.path.exists(time_file_path):
    # 如果文件存在，则读取文件
    df = pd.read_csv(time_file_path)
    df[model_name]=train_time
    df.to_csv(time_file_path, index=False)
    # print("文件已存在，正在读取...")
    # print(df)
else:
    # 如果文件不存在，则创建一个空的DataFrame，并保存为CSV文件
    df = pd.DataFrame(data=train_time,columns=[model_name])  # 根据你的实际需求定义列名
    df.to_csv(time_file_path, index=False)
    # print("文件不存在，已创建空文件.")
print('min valid loss {:5.5f}'.format(valid_loss))