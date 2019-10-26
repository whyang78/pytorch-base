import torch
from torch import nn,optim
from torch.nn import functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
from visdom import Visdom

#加入命令行操作
parser=argparse.ArgumentParser(description='rebuild mnist with vae')
parser.add_argument('--batch_size',type=int,default=32,help='dataset batch_size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--epoch',type=int,default=10,help='epoch')
parser.add_argument('--dataset_path',type=str,help='dataset path')
args=parser.parse_args()

#加载数据集
bs=args.batch_size
Epoch=args.epoch
lr=args.lr
dataset_path=args.dataset_path

train_data=datasets.MNIST(root=dataset_path,train=True,download=False,
                          transform=transforms.ToTensor()) #转换到[0,1]
trainDataLoader=torch.utils.data.DataLoader(train_data,batch_size=bs,shuffle=True)

test_data=datasets.MNIST(root=dataset_path,train=False,download=False,
                            transform=transforms.ToTensor())
testDataLoader=torch.utils.data.DataLoader(test_data,batch_size=bs,shuffle=False)

class vae(nn.Module):
    def __init__(self,z_size):
        super(vae, self).__init__()
        self.fc1=nn.Linear(28*28,512)
        self.fc2=nn.Linear(512,128)
        self.fc31=nn.Linear(128,z_size)
        self.fc32=nn.Linear(128,z_size)

        self.fc4=nn.Linear(z_size,128)
        self.fc5=nn.Linear(128,512)
        self.fc6=nn.Linear(512,28*28)

    def encoder(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        mu=self.fc31(x)
        log_var=self.fc32(x) # log(sigma**2)
        return mu,log_var

    #vae重点 ：根据分布进行采样
    def sample(self,mu,log_var):
        std=torch.exp(0.5*log_var)
        samp=std*torch.randn_like(std)+mu
        return samp

    def decoder(self,x):
        x=self.fc4(x)
        x=F.relu(x)
        x=self.fc5(x)
        x=F.relu(x)
        x=self.fc6(x)
        x=F.sigmoid(x)
        return x

    def forward(self,x):
        batchz=x.size(0)
        x=x.view(batchz,-1)
        mu,log_var=self.encoder(x)
        samp=self.sample(mu,log_var)
        out=self.decoder(samp)
        out=out.view(batchz,1,28,28)
        return out,mu,log_var

#BCE loss + KL loss
#学习的分布趋于标准正态分布
def loss_function(real_data,rebuild_data,mu,log_var,kl_coef=1.0):
    BCE_loss=F.binary_cross_entropy(rebuild_data.view(-1,784),real_data.view(-1,784),reduction='sum')
    KL_loss=-0.5*torch.sum(log_var-torch.exp(log_var)-torch.pow(mu,2)+1)
    return BCE_loss+kl_coef*KL_loss

viz=Visdom()
# viz.line([0.0],[0.0],win='train loss',opts=dict(title='train loss'))

device=torch.device('cuda:0')
net=vae(z_size=2).to(device)
optimizer=optim.Adam(net.parameters(),lr=lr)

global_step=0
for epoch in range(Epoch):
    net.train()
    for batchidx,(batchData,_) in enumerate(trainDataLoader):
        batchData= batchData.to(device)
        output,mu,log_var = net(batchData)
        loss = loss_function(batchData,output,mu,log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # global_step+=1
        # viz.line([loss.item()],[global_step],win='train loss',update='append')
    print('finish')

    #使用同一组测试集测试重建效果
    net.eval()
    iteral = iter(testDataLoader)
    batchData, _=next(iteral)
    batchData= batchData.to(device)
    with torch.no_grad():
        output, _1, _2 =net(batchData)
    viz.images(batchData,nrow=8,win='real images',opts=dict(title='real images'))
    viz.images(output,nrow=8,win='rebuild images',opts=dict(title='rebuild images'))

#利用一个符合中间分布的随机数据，经过decoder建立图像
data=torch.randn(64,2).to(device)
output=net.decoder(data)
output=output.cpu().view(64,1,28,28)
save_image(output,'./generate.jpg',nrow=8)
