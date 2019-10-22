import torch
from torch import nn,optim
from torchvision import datasets,transforms
from visdom import Visdom

bs=200
lr=0.01
Epoch=10
dataset_path='../dataset'

#划分数据集
trainData=datasets.MNIST(root=dataset_path,train=True,download=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,)),
                         ]))
trainData,valData=torch.utils.data.random_split(trainData,[50000,10000])

testData=datasets.MNIST(root=dataset_path,train=False,download=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,)),
                         ]))

trainDataLoader=torch.utils.data.DataLoader(trainData,batch_size=bs,shuffle=True)
valDataLoader=torch.utils.data.DataLoader(valData,batch_size=bs,shuffle=False)
testDataLoader=torch.utils.data.DataLoader(testData,batch_size=bs,shuffle=False)

#搭建网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model=nn.Sequential(
            nn.Linear(28*28,256),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),

            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),

            nn.Linear(64, 10),
        )

    def forward(self, x):
        x=x.view(-1,28*28)
        x=self.model(x)
        return x

device=torch.device('cuda:0')
net=MLP().to(device)
optimizer=optim.SGD(net.parameters(),lr=lr,momentum=0.9)
criterion=nn.CrossEntropyLoss().to(device)

#初始化visdom
viz=Visdom()
viz.line([0.0],[0.0],win='train_loss',opts=dict(title='train loss'))
viz.line([[0.0,0.0]],[0.0],win='valivation',opts=dict(title='loss & accuracy',legend=['loss','accuracy']))

global_step=0
for epoch in range(Epoch):
    net.train() #开启训练模式 对dropout和batchnorm层有影响
    for batch_index,(batchData,batchLabel) in enumerate(trainDataLoader):
        batchData,batchLabel=batchData.to(device),batchLabel.to(device)
        output=net(batchData)
        loss=criterion(output,batchLabel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step+=1
        viz.line([loss.item()],[global_step],win='train_loss',update='append')
        if batch_index%10==0:
            print('epoch:{},batch:{},loss:{:.4f}'.format(epoch,batch_index,loss.item()))

    net.eval() #开启测试模式 dropout层失效 验证和测试时使用完整的网络结构
    val_loss=0.0
    total_correct=0.0
    for batchData,batchLabel in valDataLoader:
        batchData,batchLabel=batchData.to(device),batchLabel.to(device)
        output=net(batchData)
        loss=criterion(output,batchLabel)

        pred=torch.argmax(output,dim=1)
        correct=torch.eq(pred,batchLabel).sum().float().item()
        total_correct+=correct
        val_loss+=loss.item()

    viz.line([[val_loss,total_correct/float(len(valData))]],[global_step],win='valivation',update='append')
    viz.images(batchData,win='imageSample')
    viz.text(str(pred.detach().cpu().numpy()),win='prediction',opts=dict(title='pred'))

net.eval()  # 开启测试模式 dropout层失效 验证和测试时使用完整的网络结构
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
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, total_correct, len(testDataLoader.dataset),
    100. * total_correct / len(testDataLoader.dataset)))