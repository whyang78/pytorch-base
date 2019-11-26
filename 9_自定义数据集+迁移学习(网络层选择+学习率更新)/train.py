import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms
from MyUtils import MyResnet18,mydataset
from visdom import Visdom

dataset_path='../dataset/pokemon'
bs=32
lr=0.001
Epoch=20

#定义随机数种子
torch.manual_seed(78)

#构造数据集
trainData=mydataset(dataset_path,mode='train',transform=transforms.Compose([
                transforms.Resize((224,224)), #直接两边转换为224,224
                # transforms.Resize(224), #保持纵横比不变，最短边为224
                # transforms.CenterCrop(224), #中心裁剪224*224的图像
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

valData=mydataset(dataset_path,mode='val',transform=transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

testData=mydataset(dataset_path,mode='test',transform=transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

trainDataLoader=DataLoader(trainData,batch_size=bs,shuffle=True)
testDataLoader=DataLoader(testData,batch_size=bs,shuffle=False)
valDataLoader=DataLoader(valData,batch_size=bs,shuffle=False)

#可视化
viz=Visdom()
viz.line([0.0],[0.0],win='train_loss',opts=dict(title='train_loss'))
viz.line([0.0],[0.0],win='val_accuracy',opts=dict(title='val_accuracy'))

#构建网络
device=torch.device('cuda:0')
net=MyResnet18(5).to(device) #一共5类
optimizer=optim.Adam(net.parameters(),lr=lr,weight_decay=0.001)
criterion=nn.CrossEntropyLoss().to(device)

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
        torch.save(net.state_dict(),'./best.mdl')

print('val best accuracy:',best_accuracy)

#加载网络参数
net.load_state_dict(torch.load('./best.mdl'))
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
