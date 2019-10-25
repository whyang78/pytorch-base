import torch
import numpy as np
from torch import nn,optim
import pandas as pd
import matplotlib.pyplot as plt

#处理数据
data=pd.read_csv('./LSTM_time_data.csv',usecols=[1])
data=data.dropna()
dataset=data.values.astype('float32')
data_max,data_min=np.max(dataset),np.min(dataset)
dataset=(dataset-data_min)/(data_max-data_min) #归一化

def create_dataset(dataset,lookback=2):
    data_x,data_y=[],[]
    for i in range(len(dataset)-lookback):
        data=dataset[i:(i+lookback)]
        data_x.append(data)
        data_y.append(dataset[i+lookback])
    return np.array(data_x),np.array(data_y)

#划分数据集
data_x,data_y=create_dataset(dataset)
train_x=data_x[:int(len(data_x)*0.7)]
train_y=data_y[:int(len(data_x)*0.7)]
test_x=data_x[int(len(data_x)*0.7):]
test_y=data_y[int(len(data_x)*0.7):]

train_x = train_x.reshape(-1, 1, 2)
train_y = train_y.reshape(-1, 1, 1)
test_x = test_x.reshape(-1, 1, 2)
test_y = test_y.reshape(-1, 1, 1)
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)

#构建网络
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.lstm=nn.LSTM(input_size=2,hidden_size=10,num_layers=1)
        self.fc=nn.Linear(10,1)

    def forward(self, x):
        x,_=self.lstm(x)
        s,b,f=x.shape
        x=x.view(-1,f)
        x=self.fc(x)
        x=x.view(s,b,-1)
        return x

net=net()
optimizer=optim.Adam(net.parameters(),lr=0.01)
criterion=nn.MSELoss()

#训练
for epoch in range(1000):
    pred=net(train_x)
    loss=criterion(pred,train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%10==0:
        print('epoch:{},loss:{:.6f}'.format(epoch+1,loss.data))

#测试
pred=net(test_x)
pred=pred.data.float().numpy().ravel()
test_y=test_y.data.float().numpy().ravel()
plt.plot(pred,color='r',label='prediction')
plt.plot(test_y,color='g',label='true')
plt.show()