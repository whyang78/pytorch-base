import torch
from torch import nn,optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from visdom import Visdom

#加载数据集
bs=32
Epoch=10
lr=0.01
dataset_path='../dataset'

train_data=datasets.MNIST(root=dataset_path,train=True,download=False,
                          transform=transforms.ToTensor()) #转换到[0,1]
trainDataLoader=torch.utils.data.DataLoader(train_data,batch_size=bs,shuffle=True)

test_data=datasets.MNIST(root=dataset_path,train=False,download=False,
                            transform=transforms.ToTensor())
testDataLoader=torch.utils.data.DataLoader(test_data,batch_size=bs,shuffle=False)

#构建网络
#全连接层AE
class autoEncoder(nn.Module):
    def __init__(self,input_size,hidden_size_1,hidden_size_2,z_size):
        super(autoEncoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_size,hidden_size_1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size_1,hidden_size_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size_2,z_size),
        )

        self.decoder=nn.Sequential(
            nn.Linear(z_size,hidden_size_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size_2,hidden_size_1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size_1,input_size),
            nn.Sigmoid(), #将数值转换到[0,1]
        )

    def forward(self, x):
        batch_size=x.size(0)
        x=x.view(-1,28*28)
        x=self.encoder(x)
        x=self.decoder(x)
        x=x.view(batch_size,1,28,28)
        return x

#卷积AE
class convAutoEncoder(nn.Module):
    def __init__(self):
        super(convAutoEncoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=2,padding=1), # 8,14,14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), #8,7,7

            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1), # 16,7,7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2) #16,3,3
        )

        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(in_channels=16,out_channels=8,kernel_size=3,stride=2,padding=0), # 8,7,7
            # stride*(width-1)-2*padding+kernel
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8,out_channels=8,kernel_size=2,stride=2,padding=0),# 8,14,14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8,out_channels=1,kernel_size=2,stride=2,padding=0),# 1,28,28
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

viz=Visdom()
viz.line([0.0],[0.0],win='train loss',opts=dict(title='train loss'))

device=torch.device('cuda:0')
net=autoEncoder(input_size=28*28,hidden_size_1=256,hidden_size_2=128,z_size=4).to(device)
# net=convAutoEncoder().to(device)
optimizer=optim.Adam(net.parameters(),lr=lr)
criterion=nn.MSELoss().to(device)

global_step=0
iteral=iter(testDataLoader)
for epoch in range(Epoch):
    net.train()
    for batchidx,(batchData,_) in enumerate(trainDataLoader):
        batchData= batchData.to(device)
        output = net(batchData)  # ->tensor cuda
        loss = criterion(output, batchData)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step+=1
        viz.line([loss.item()],[global_step],win='train loss',update='append')
    print('finish')

    net.eval()
    batchData, _=next(iteral)
    batchData= batchData.to(device)
    with torch.no_grad():
        output=net(batchData)
    viz.images(batchData,nrow=8,win='real images',opts=dict(title='real images'))
    viz.images(output,nrow=8,win='rebuild images',opts=dict(title='rebuild images'))



