import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

dataset_path='../dataset'
bs=5
lr=0.001
Epoch=1

#加载数据集
train_data=datasets.MNIST(root=dataset_path,train=True,download=False,
                                      transform=transforms.Compose(
                                          [
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,)),
                                          ]
                                      ))
trainDataLoader=torch.utils.data.DataLoader(train_data,batch_size=bs,shuffle=True)

test_data=datasets.MNIST(root=dataset_path,train=False,download=False,
                                     transform=transforms.Compose(
                                         [
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,)),
                                         ]
                                     ))
testDataLoader=torch.utils.data.DataLoader(test_data,batch_size=bs,shuffle=False)

class net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=10,num_layers=2):
        super(net, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x,_=self.lstm(x)
        x=x[:,-1,:] #选出序列的最后一个输出
        x=self.fc(x)
        return x

device=torch.device('cuda:0')
net=net(28,64).to(device)
optimizer=optim.Adam(net.parameters(),lr=lr)
criterion=nn.CrossEntropyLoss().to(device)

for epoch in range(Epoch):
    for batchidx,(batchData,batchLabel) in enumerate(trainDataLoader):
        batchData=batchData.squeeze()
        batchData, batchLabel=batchData.to(device),batchLabel.to(device)
        output=net(batchData)
        loss=criterion(output,batchLabel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batchidx%100==0:
            print('epoch:{},batch:{},loss:{}'.format(epoch,batchidx,loss.data))

#测试网络，并绘制混淆矩阵
correct=0.0
total=0.0
for step,(data,label) in enumerate(testDataLoader):
    data=data.squeeze()
    data, label=data.to(device),label.to(device)
    output=net(data)
    prediction=torch.argmax(output,dim=1)
    correct+=torch.eq(prediction,label).sum().float().item()
    total+=label.size(0)
    print('已测试数目:',total)

accuracy=float(correct)/float(total)
print('准确率:',accuracy)
