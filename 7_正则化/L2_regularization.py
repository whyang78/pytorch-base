import torch
from torch import nn,optim
from torch.nn import functional as F
import torchvision
import os
import time

start=time.time()
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
net=Net()
optimizer=optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=0.01) #L2正则化
criterion=nn.CrossEntropyLoss()

train_loss=[]
for epoch in range(Epoch):
    for batchIndex,(batchData,batchLabel) in enumerate(trainDataLoader):
        output=net(batchData)
        loss=criterion(output,batchLabel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        if batchIndex%10==0:
            print(epoch,batchIndex,loss.item())

#测试网络
total_correct=0.0
for batchData,batchLabel in testDataLoader:
    output=net(batchData)
    prediction=torch.argmax(output,dim=1) #->tensor
    correct=torch.eq(prediction,batchLabel).sum().float().item() #->float data
    total_correct+=correct
print('test accuracy:',total_correct/float(len(testDataLoader.dataset)))




