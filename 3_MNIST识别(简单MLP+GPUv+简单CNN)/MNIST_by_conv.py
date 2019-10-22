import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as f
import torchvision
import matplotlib.pyplot as plt
import os

download=False
bs=64
lr=0.01
Epoch=3

#加载数据集
if not os.path.exists('../dataset') or not os.listdir('../dataset'):
    download=True

train_data=torchvision.datasets.MNIST(root='../dataset',train=True,download=download,
                                      transform=torchvision.transforms.Compose(
                                          [
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                          ]
                                      ))
trainDataLoader=torch.utils.data.DataLoader(train_data,batch_size=bs,shuffle=True)

test_data=torchvision.datasets.MNIST(root='../dataset',train=False,download=False,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                         ]
                                     ))
testDataLoader=torch.utils.data.DataLoader(test_data,batch_size=bs,shuffle=False)

#构建网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #1*28*28，其实是四维的量，因为是按批次训练(train_loader)，实际维度为[batch_size,1,28,28]，下面的都是如此
        self.conv1=nn.Sequential(
            nn.Conv2d( in_channels=1, out_channels=16,kernel_size=5,stride=1,padding=2),#16*28*28
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)#16*14*14
        )
        #16*14*14
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2),#32*14*14
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)#32*7*7
        )
        #需要展平
        self.out=nn.Sequential(
            nn.Linear(in_features=32*7*7,out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100,out_features=10)
        )

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        x=self.out(x)
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

#训练网络
device=torch.device('cuda:0') #选择cuda
net=CNN().to(device) #网络在指定cuda上
net.initialize_weights()# 权值初始化
optimizer=optim.SGD(net.parameters(),lr=lr,momentum=0.9)
criterion=nn.CrossEntropyLoss().to(device) #损失函数在cuda上运算

train_loss=[]
for epoch in range(Epoch):
    for batchIndex,(batchData,batchLabel) in enumerate(trainDataLoader):
        batchData,batchLabel=batchData.to(device),batchLabel.to(device)
        output=net(batchData) #->tensor cuda
        loss=criterion(output,batchLabel) #标签需要转换成cuda类型

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        if batchIndex%10==0:
            print(epoch,batchIndex,loss.item())

#测试网络
total_correct=0.0
for batchData,batchLabel in testDataLoader:
    batchData, batchLabel = batchData.to(device), batchLabel.to(device)
    output=net(batchData)
    prediction=torch.argmax(output,dim=1)
    correct=torch.eq(prediction,batchLabel).sum().float().item()
    total_correct+=correct
print('test accuracy:',total_correct/float(len(testDataLoader.dataset)))


