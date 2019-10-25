import torch
from torch import nn,optim
import numpy as np
import matplotlib.pyplot as plt

batch_size=1
input_size=1
hidden_size=10
output_size=1
numSteps=50
lr=0.01

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.rnn=nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x,h=self.rnn(x)
        b,s,f=x.shape
        x=x.view(-1,f)
        x=self.fc(x)
        x=x.view(b,s,-1)
        return x

    def initial_params(self):
        for p in self.rnn.parameters():
            nn.init.normal_(p,0,0.001)

device=torch.device('cpu')
rnn=net()
rnn.initial_params()
rnn.to(device)
optimizer=optim.SGD(rnn.parameters(),lr=lr,momentum=0.9)
criterion=nn.MSELoss().to(device)

for epoch in range(5000):
    #初始化数据
    start=np.random.randint(5,size=1)[0]
    time_steps = np.linspace(start, start + 10, numSteps)
    data = np.sin(time_steps)
    data = data.reshape(numSteps, 1)
    x = torch.tensor(data[:-10]).float().view(1, numSteps - 10, 1) #[batch,seq,feature]
    y = torch.tensor(data[10:]).float().view(1, numSteps - 10, 1)

    x,y=x.to(device),y.to(device)
    pred=rnn(x)
    loss=criterion(pred,y)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(),max_norm=10,norm_type=2) #梯度裁剪
    optimizer.step()

    if epoch%10==0:
        print('epoch:{},loss:{:.6f}'.format(epoch+1,loss.data))

start=np.random.randint(5,size=1)[0]
time_steps = np.linspace(start, start + 10, numSteps)
data = np.sin(time_steps)
data = data.reshape(numSteps, 1)
x = torch.tensor(data[:-10]).float().view(1, numSteps - 10, 1) #[batch,seq,feature]
y = torch.tensor(data[10:]).float().view(1, numSteps - 10, 1)

#预测
pred=rnn(x)
pred=pred.data.float().numpy().ravel()

x=x.data.float().numpy().ravel()
y=y.data.float().numpy().ravel()
plt.scatter(time_steps[10:],y,c='r',marker='o')
plt.plot(time_steps[10:],y,color='g')
plt.scatter(time_steps[10:],pred,c='b',marker='*')
plt.show()

