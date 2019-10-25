import torch
from torch import nn

if __name__ == '__main__':
    #RNN使用
    rnn=nn.RNN(input_size=100,hidden_size=30,num_layers=1)
    x = torch.randn(10, 3, 100)  # [seq,batch,feature]
    h = torch.zeros(1,3,30) #[num_layers,batch,hidden_feature]
    x,h=rnn(x,h)
    print(x.shape,h.shape)

    #RNN layers使用  其实是纵向的设置
    #single layer
    layer=nn.RNNCell(input_size=100,hidden_size=30)
    x = torch.randn(10, 3, 100)
    h=torch.zeros(3,30)
    for xi in x:
        h=layer(xi,h)
    print(h.shape)

    #two layers
    layer1=nn.RNNCell(input_size=100,hidden_size=30)
    layer2=nn.RNNCell(input_size=30,hidden_size=20)
    x = torch.randn(10, 3, 100)
    h1=torch.zeros(3,30)
    h2=torch.zeros(3,20)
    for xi in x:
        h1=layer1(xi,h1)
        h2=layer2(h1,h2)
    print(h1.shape,h2.shape)

    #lstm
    lstm=nn.LSTM(input_size=100,hidden_size=30,num_layers=1)
    x = torch.randn(10, 3, 100)  # [seq,batch,feature]
    h = torch.zeros(1,3,30) #[num_layers,batch,hidden_feature]
    c=torch.zeros(1,3,30)
    x,(h,c)=lstm(x,(h,c))
    print(x.shape,h.shape,c.shape)

    #lstm layers
    #single layer
    x = torch.randn(10, 3, 100)
    layer=nn.LSTMCell(input_size=100,hidden_size=30)
    h = torch.zeros(3,30)
    c=torch.zeros(3,30)
    for xi in x:
        h,c=layer(xi,(h,c))
    print(h.shape,c.shape)

    #two layers
    x = torch.randn(10, 3, 100)
    layer1=nn.LSTMCell(input_size=100,hidden_size=30)
    layer2=nn.LSTMCell(input_size=30,hidden_size=20)
    h1 = torch.zeros(3,30)
    h2=torch.zeros(3,20)
    c1=torch.zeros(3,30)
    c2=torch.zeros(3,20)
    for xi in x:
        h1,c1=layer1(xi,(h1,c1))
        h2,c2=layer2(h1,(h2,c2))
    print(h2.shape,c2.shape)