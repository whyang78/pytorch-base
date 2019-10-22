import torch
from torch import nn,optim
from torch.nn import functional as F
from torchvision import transforms,datasets

bs=64
Epoch=2
lr=0.01
dataset_path='../dataset'

train_data=datasets.MNIST(root=dataset_path,train=True,download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))
trainDataLoader=torch.utils.data.DataLoader(train_data,batch_size=bs,shuffle=True)

test_data=datasets.MNIST(root=dataset_path,train=False,download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))
testDataLoader=torch.utils.data.DataLoader(test_data,batch_size=bs,shuffle=False)

#参数的初始化
w1,b1=torch.randn(256,28*28,requires_grad=True), torch.zeros(256,requires_grad=True)
w2,b2=torch.randn(64,256,requires_grad=True), torch.zeros(64,requires_grad=True)
w3,b3=torch.randn(10,64,requires_grad=True), torch.zeros(10,requires_grad=True)
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

def forward(x):
    x=x.view(-1,28*28)
    x=x@w1.t()+b1
    x=F.relu(x)
    x=x@w2.t()+b2
    x=F.relu(x)
    x=x@w3.t()+b3

    return x

optimizer=optim.SGD([w1,b1,w2,b2,w3,b3],lr=lr)
criterion=nn.CrossEntropyLoss()

for epoch in range(Epoch):
    for batch_index,(batch_data,batch_label) in enumerate(trainDataLoader):
        output=forward(batch_data)
        loss=criterion(output,batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_loss=0.0
total_correct=0
for batch_data, batch_label in testDataLoader:
    output = forward(batch_data)
    loss = criterion(output, batch_label)

    pred=torch.argmax(output,dim=1)
    correct=torch.eq(pred,batch_label).sum().float().item()
    total_correct+=correct
    test_loss+=loss.item()
print('{}/{},loss:{:.2f}'.format(total_correct,len(testDataLoader.dataset),test_loss))

