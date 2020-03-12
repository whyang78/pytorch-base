import torch
from torch import nn
from torch.nn import functional as F

#残差模块
class ResidualBlock(nn.Module):
    def __init__(self,inChannel,outChannel,stride=1):
        super(ResidualBlock, self).__init__()

        self.left=nn.Sequential(
            nn.Conv2d(inChannel,outChannel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outChannel),
            nn.Conv2d(outChannel,outChannel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
        )

        self.short_cut=nn.Sequential()
        if stride!=1 or inChannel!=outChannel:
            self.short_cut=nn.Sequential(
                nn.Conv2d(inChannel,outChannel,kernel_size=1,stride=stride,padding=0,bias=False),
                nn.BatchNorm2d(outChannel),
            )

    def forward(self, x):
        out=self.left(x)
        out+=self.short_cut(x)
        out=F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self,Block,num_classes=10):
        super(ResNet18, self).__init__()
        #预处理3*32*32 -> 64*32*32
        self.conv=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.inChannel=64

        self.layer1=self.make_layers(Block,64,2,1) #64*32*32 -> 64*32*32
        self.layer2=self.make_layers(Block,128,2,2) #64*32*32 -> 128*16*16
        self.layer3 = self.make_layers(Block, 256, 2, 2) #128*16*16 -> 256*8*8
        self.layer4 = self.make_layers(Block, 512, 2, 2) #256*8*8 -> 512*4*4
        self.fc=nn.Linear(512,10)

    def make_layers(self,Block,outChannel,num_block,stride):
        strides=[stride]+[1]*(num_block-1)
        blocks=[]
        for s in strides:
            blocks.append(Block(self.inChannel,outChannel,s))
            self.inChannel=outChannel
        return nn.Sequential(*blocks)

    def forward(self, x):
        x=self.conv(x)
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=F.adaptive_avg_pool2d(x, [1, 1]) #512*1*1
        x=x.view(-1,512)
        x=self.fc(x)
        return x

def MyResnet18():
    return ResNet18(ResidualBlock)