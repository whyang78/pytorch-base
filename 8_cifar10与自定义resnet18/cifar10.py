import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from visdom import Visdom
from Myresnet import MyResnet18

bs=50
lr=0.001
Epoch=20
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
dataset_path='../dataset/cifar'

#加载数据集
trainData=datasets.CIFAR10(root=dataset_path,train=True,download=False,
                           transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))
testData=datasets.CIFAR10(root=dataset_path,train=False,download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))
trainDataLoader=DataLoader(trainData,batch_size=bs,shuffle=True)
testDataLoader=DataLoader(testData,batch_size=bs,shuffle=False)

#定义可视化
viz=Visdom()
viz.line([0.0],[0.0],win='train_loss',opts=dict(title='train_loss'))
viz.line([0.0],[0.0],win='test_loss',opts=dict(title='test_loss'))
viz.line([0.0],[0.0],win='test_accuracy',opts=dict(title='test_accuracy'))

#加载网络
device=torch.device('cuda:0') #
net=MyResnet18().to(device)
optimizer=optim.Adam(net.parameters(),lr=lr,weight_decay=5e-4)
criterion=nn.CrossEntropyLoss().to(device)

global_step=0
for epoch in range(Epoch):
    net.train()
    for batch_index,(batchData,batchLabel) in enumerate(testDataLoader):
        batchData, batchLabel=batchData.to(device),batchLabel.to(device)
        output=net(batchData)
        loss=criterion(output,batchLabel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step+=1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')
        # if batch_index%20==0:
        #     print('epoch:{},batch:{},loss:{:.4f}'.format(epoch,batch_index,loss.item()))

    net.eval()
    with torch.no_grad():
        test_loss = 0.0
        total_correct = 0.0
        for batchData, batchLabel in testDataLoader:
            batchData, batchLabel = batchData.to(device), batchLabel.to(device)
            output = net(batchData)
            loss = criterion(output, batchLabel)

            pred = torch.argmax(output, dim=1)
            correct = torch.eq(pred, batchLabel).sum().float().item()
            total_correct += correct
            test_loss += loss.item()

    test_loss /= len(testDataLoader.dataset)
    accuracy=total_correct / len(testDataLoader.dataset)
    viz.line([test_loss], [epoch+1], win='test_loss', update='append')
    viz.line([accuracy], [epoch+1], win='test_accuracy', update='append')

    print('\nepoch: {},Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch,test_loss, total_correct, len(testDataLoader.dataset),
        100. * total_correct / len(testDataLoader.dataset)))



#显示每一类的准确率
# classes_correct=list(0 for i in range(10))
# classes_total=list(0 for i in range(10))
# for data in testloader:
#     images,labels=data
#     outputs=net(images)
#     prediction = torch.max(outputs.data, 1)[1]
#     correct += (prediction == labels).squeeze()
#     for j in range(batch_size):
#         label=labels[j]
#         classes_correct[label]+=correct[j]
#         classes_total[label]+=1
# for i in range(10):
#     print('accuracy of %5s :%.2f %%'%(classes[i],100*float(classes_correct[i])/float(classes_total[i])))