import torch
from torch import nn,optim
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms
from torchvision.models import resnet18
from visdom import Visdom

dataset_path='../dataset/pokemon'
bs=32
lr=0.001
Epoch=20
torch.manual_seed(78)

#使用imagefolder加载数据集
#其实在这个地方不太适合用这个，因为训练集验证集和测试集还没有划分，且他们的tf不一样，若是各个集分开存放，则比较适合
tf=transform=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

dataset=datasets.ImageFolder(root=dataset_path,transform=tf)

n=int(len(dataset)*0.2)
trainData,testData=random_split(dataset,[len(dataset)-n,n])
trainData,valData=random_split(trainData,[len(trainData)-n,n])

trainDataLoader=DataLoader(trainData,batch_size=bs,shuffle=True)
testDataLoader=DataLoader(testData,batch_size=bs,shuffle=False)
valDataLoader=DataLoader(valData,batch_size=bs,shuffle=False)

#构建网络
device=torch.device('cuda:0')
net=resnet18(pretrained=True) #若是要改变net参数，要先改变，然后再device，否则新改变的层的参数不会转移到device上
criterion=nn.CrossEntropyLoss().to(device)

fix=True #是否冻结网络，即网络参数是否学习，一般情况下只学习最后一层全连接即可，其它层不用学习
if fix:
    for param in net.parameters():
        param.requires_grad=False
net.fc=nn.Linear(net.fc.in_features,5) #5类输出
net=net.to(device)

if fix:
    optimizer=optim.Adam(net.fc.parameters(),lr=lr)
else:
    optimizer=optim.Adam(net.parameters(),lr=lr)
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1) #更新学习率

#可视化
viz=Visdom()
viz.line([0.0],[0.0],win='train_loss',opts=dict(title='train_loss'))
viz.line([0.0],[0.0],win='val_accuracy',opts=dict(title='val_accuracy'))

best_accuracy=0.0
global_step=0
for epoch in range(Epoch):
    net.train()
    for batchData, batchLabel in trainDataLoader:
        batchData, batchLabel = batchData.to(device), batchLabel.to(device)
        output = net(batchData)
        loss = criterion(output, batchLabel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step+=1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')
    print(scheduler.get_lr())

    net.eval()
    with torch.no_grad():
        total_correct=0.0
        for batchData, batchLabel in valDataLoader:
            batchData, batchLabel = batchData.to(device), batchLabel.to(device)
            output = net(batchData)

            pred=torch.argmax(output,dim=1)
            correct=torch.eq(pred,batchLabel).sum().float().item()
            total_correct+=correct

    accuracy=total_correct/float(len(valData))
    viz.line([accuracy], [epoch + 1], win='val_accuracy', update='append')
    #保存准确率最好的网络参数
    if accuracy>best_accuracy:
        best_accuracy=accuracy
        torch.save(net.state_dict(),'./transfer_best.mdl')

    scheduler.step()  # 更新学习率

print('val best accuracy:',best_accuracy)

#加载网络参数
net.load_state_dict(torch.load('./transfer_best.mdl'))
net.eval()
with torch.no_grad():
    total_correct = 0.0
    for batchData, batchLabel in testDataLoader:
        batchData, batchLabel = batchData.to(device), batchLabel.to(device)
        output = net(batchData)

        pred = torch.argmax(output, dim=1)
        correct = torch.eq(pred, batchLabel).sum().float().item()
        total_correct += correct
accuracy = total_correct / float(len(valData))
print('test accuracy:',accuracy)
