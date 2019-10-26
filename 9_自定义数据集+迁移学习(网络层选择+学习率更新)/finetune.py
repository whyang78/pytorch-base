import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from PIL import Image
from torchvision import transforms,datasets
import os
from Confusion_Matrix import test_confmat,show_confmat #sys.path已包含此py程序所在路径

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
download_enable=False
LR=0.001
batch_size=5
EPOCH=1

#--------------------------------制作数据集-----------------------------------#
class mydataset(Data.Dataset):
    def __init__(self,file_path,transforms=None,target_transforms=None):
        super(mydataset, self).__init__()
        img=[]
        with open(file_path,'r') as f:
            for line in f:
                line=line.rstrip()
                words=line.split()
                (data_path, label)=(words[0],int(words[1]))
                img.append((data_path,label))
        self.img=img
        self.transforms=transforms
        self.target_transforms=target_transforms

    def __getitem__(self, index):
        data_path,label=self.img[index]
        data=Image.open(data_path).convert('RGB')

        if self.transforms is not None:
            data=self.transforms(data)

        return  data,label

    def __len__(self):
        return len(self.img)

#--------------------------------制作迭代器-----------------------------#
train_txt_path='./cifar/data/my_data/train_data.txt'
test_txt_path='./cifar/data/my_data/test_data.txt'

norm_mean=[0.4937733, 0.48471925, 0.4501275]
norm_std=[0.24644545, 0.24266596, 0.26123333]
normtransform=transforms.Normalize(norm_mean,norm_std)
train_transform=transforms.Compose(
    [
        transforms.Resize(32),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        normtransform
    ]
)
test_transform=transforms.Compose(
    [
        transforms.ToTensor(),
        normtransform
    ]
)
train_data=mydataset(file_path=train_txt_path,transforms=train_transform)
test_data=mydataset(file_path=test_txt_path,transforms=test_transform)

train_loader=Data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader=Data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

# loader=iter(train_loader)
# img,label=next(loader)
# print(img.size(),label)

#---------------------------------配置网络------------------------------#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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

net=Net()
net.initialize_weights()
#---------------------------导入网络参数，配置学习率 finetune----------------------#
pretrain_dict=torch.load('mydataset_net_params.pkl')
net_dict=net.state_dict()

pretrain_dict_1={k:v for k,v in pretrain_dict.items() if k in net_dict}
net_dict.update(pretrain_dict_1)
net.load_state_dict(net_dict)

ignored_params=list(map(id,net.fc3.parameters()))
base_params=filter(lambda p:id(p) not in ignored_params,net.parameters())

#配置优化器和损失函数
optimizer=optim.Adam([
    {'params':base_params},
    {'params':net.fc3.parameters(),'lr':LR*10}
],lr=LR,betas=(0.9,0.99))

loss_func=nn.CrossEntropyLoss()
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)#更新学习率


#-----------------------------------训练网络---------------------------------#
count=0
for epoch in range(EPOCH):
    scheduler.step()
    for step,(batch_data,batch_label) in enumerate(train_loader):
        prediction=net(batch_data)
        loss=loss_func(prediction,batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count+=batch_label.size(0)
        print('批次:',epoch,' 已训练数目:',count)

#---------------------------------绘制混淆矩阵-----------------------------------#
output_path='./cifar/data/my_data/'
confmat,accuracy=test_confmat(net,test_loader,'test',classes)
print('accuracy:{}'.format(accuracy))
show_confmat(confmat,classes,'test',output_path)
