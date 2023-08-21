import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, input_size=10, input_window=7, feature_size=250, num_layers=1, dropout=0.0):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size * input_window, 1)
        self.hidden_size = feature_size
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.rnn_num_layers = num_layers
        self.init_linear = nn.Linear(input_size, feature_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.init_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.view(output.size(0), -1)
        output = self.decoder(output)
        output.view(output.size(0))
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_lstm_state(self, batch_size):
        return torch.zeros((self.rnn_num_layers, batch_size, self.hidden_size), device=self.device)

class MLP(nn.Module):

    def __init__(self, input_size=10, hidden_size=32, input_window=7 , output_size=1 , dropout=0, batch_first=False,device=torch.device('cpu')):
        super(MLP, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        #self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.input_size*input_window, self.output_size)

    def forward(self,x):
        x=x.view(x.size(0),-1)
        out = self.linear(x)
        #out=out.view(out.size(0),-1)
        return out

class MLP2(nn.Module):

    def __init__(self, input_size=10, hidden_size=32, input_window=7 , output_size=1 , dropout=0, batch_first=False,device=torch.device('cpu')):
        super(MLP2, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        # self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.input_size*input_window, self.hidden_size)
        #self.ln = nn.BatchNorm1d(self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,x):
        x=x.view(x.size(0),-1)
        out = self.linear(x)
        out = F.relu(out)
        out = self.linear2(out)
        return out

class MLP3(nn.Module):

    def __init__(self, input_size=10, hidden_size=32, input_window=7 , output_size=1 , dropout=0, batch_first=False,device=torch.device('cpu')):
        super(MLP3, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        #self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.input_size*input_window, self.hidden_size)
        self.bn1=nn.LayerNorm(self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn2 = nn.LayerNorm(self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out

class LSTM(nn.Module):

    def __init__(self, input_size=10, hidden_size=32, num_layers=1 , output_size=1 , dropout=0, batch_first=True,device=torch.device('cpu')):
        super(LSTM, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.device = device
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,x):
        hidden, cell = self.init_lstm_state(x.size(0))
        out, (hidden, cell) = self.rnn(x,(hidden, cell))  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # print(out.shape)
        # out = self.linear(hidden.reshape(a * b, c))
        hidden,cell=hidden.squeeze(0),cell.squeeze(0)
        out = self.linear(hidden)
        return out

    def init_lstm_state(self, batch_size):
        # 初始化输入参数
        #print(batch_size)
        return torch.randn((1,batch_size, self.hidden_size), device=self.device),torch.randn((1,batch_size, self.hidden_size), device=self.device)

class GRU(nn.Module):

    def __init__(self, input_size=10, hidden_size=32, num_layers=1 , output_size=1 , dropout=0, batch_first=True,device=torch.device('cpu')):
        super(GRU, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.device=device
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,x):
        hidden=self.init_lstm_state(x.size(0))
        out, hidden = self.rnn(x,hidden)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # print(out.shape)
        # out = self.linear(hidden.reshape(a * b, c))
        hidden=hidden.squeeze(0)
        out = self.linear(hidden)
        return out

    def init_lstm_state(self, batch_size):
        # 初始化输入参数
        #print(batch_size)
        return torch.randn((1,batch_size, self.hidden_size), device=self.device)

class RNN(nn.Module):

    def __init__(self, input_size=10, hidden_size=32, num_layers=1 , output_size=1 , dropout=0, batch_first=True,device=torch.device('cpu')):
        super(RNN, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.device=device
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,x):
        hidden=self.init_lstm_state(x.size(0))
        out, hidden = self.rnn(x,hidden)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # print(out.shape)
        # out = self.linear(hidden.reshape(a * b, c))
        hidden=hidden.squeeze(0)
        out = self.linear(hidden)
        return out

    def init_lstm_state(self, batch_size):
        # 初始化输入参数
        #print(batch_size)
        return torch.randn((1,batch_size, self.hidden_size), device=self.device)