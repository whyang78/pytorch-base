import torch
from torch import nn,optim
from torch.nn import functional as F
import torchvision
import os
from func import plot_image, plot_curve
import time


start=time.time()
download=False
bs=64
lr=0.01
Epoch=3
print(torch.cuda.is_available())

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

x,y=next(iter(trainDataLoader))
plot_image(x,y,'trainSample')

#搭建网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1=nn.Linear(28*28,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        x=self.fc3(x)

        return x

#训练网络
device=torch.device('cuda:0') #选择cuda
net=Net().to(device) #网络在指定cuda上
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
plot_curve(train_loss)

#测试网络
total_correct=0.0
for batchData,batchLabel in testDataLoader:
    batchData, batchLabel = batchData.to(device), batchLabel.to(device)
    output=net(batchData)
    prediction=torch.argmax(output,dim=1)
    correct=torch.eq(prediction,batchLabel).sum().float().item()
    total_correct+=correct
print('test accuracy:',total_correct/float(len(testDataLoader.dataset)))

finish=time.time()
print(finish-start)


