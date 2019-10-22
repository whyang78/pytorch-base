import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np
import glob
import os
import csv
import random
from PIL import Image

dataset_path='../../dataset/pokemon'

#整理数据集 数据集包含多种格式，全部转换成jpg格式
def clear_picture(root):
    '''
    :param root:数据集路径
    :return:
    '''
    for parent, dirname, filename in os.walk(root):
        if not filename:
            continue
        for pic in filename:
            if pic.endswith('.jpg'):
                continue
            name=pic.split('.')[0]
            new_name=name+'.jpg'
            os.rename(os.path.join(parent,pic),os.path.join(parent,new_name))
# clear_picture(dataset_path)

#建立dataset
class mydataset(Dataset):
    def __init__(self,root,mode='all',transform=None,target_transform=None):
        '''
        :param root: 数据集路径
        :param mode: all train val test
        :param transform:
        :param target_transform:
        '''
        super(mydataset, self).__init__()

        self.root=root
        self.transform=transform
        self.target_transform=target_transform

        self.name2label={}
        for name in sorted(os.listdir(root)):
            self.name2label[name]=len(self.name2label.keys())

        csv_path = 'data.csv'
        self.images,self.labels=self.load_csv(csv_path)
        if mode=='train':
            self.images=self.images[:int(len(self.images)*0.6)]
            self.labels=self.labels[:int(len(self.labels)*0.6)]
        elif mode=='val':
            self.images = self.images[int(len(self.images) * 0.6):int(len(self.images) * 0.8)]
            self.labels = self.labels[int(len(self.labels) * 0.6):int(len(self.labels) * 0.8)]
        elif mode=='test':
            self.images = self.images[int(len(self.images) * 0.8):]
            self.labels = self.labels[int(len(self.labels) * 0.8):]

    def load_csv(self,csv_path):
        '''
        :param csv_path:csv_file路径
        :return: 图片路径list label_list
        '''
        if not os.path.exists(csv_path):
            images=[]
            for subdir in os.listdir(self.root):
                images.extend(glob.glob(os.path.join(self.root,subdir,'*.jpg')))
            print(len(images))

            random.shuffle(images)
            with open(csv_path,'w+',newline='') as f:
                writer=csv.writer(f)
                for img in images:
                    name=img.split(os.sep)[-2]
                    label=self.name2label[name]
                    writer.writerow([img,label])
            print('csv_file has written!')

        images,labels=[],[]
        with open(csv_path) as f:
            reader=csv.reader(f)
            for row in reader:
                img,lab=row
                images.append(img)
                labels.append(int(lab))
        print('csv_file load!')

        assert len(images)==len(labels)
        return images,labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image=self.images[index]
        label=self.labels[index]

        image=Image.open(image).convert('RGB')
        if self.transform is not None:
            image=self.transform(image)
        label=torch.tensor(label).type(torch.LongTensor)
        return image,label

if __name__ == '__main__':
    #get标准化参数
    data=mydataset(dataset_path,mode='val',transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()]))
    print(len(data))
    # dataloader=DataLoader(data,len(data),shuffle=True)
    # imgs,labels=next(iter(dataloader))
    #
    # imgs_R = imgs[:,0,:,:].numpy().ravel()
    # imgs_G = imgs[:, 1, :, :].numpy().ravel()
    # imgs_B = imgs[:, 2, :, :].numpy().ravel()
    # print(imgs_R.shape)
    # print(1168*224*224)
    #
    # mean=[np.mean(imgs_R),np.mean(imgs_G),np.mean(imgs_B)]
    # std=[np.std(imgs_R),np.std(imgs_G),np.std(imgs_B)]
    # print(mean)
    # print(std)
    # [0.6058678, 0.6085033, 0.5339738]
    # [0.38046822, 0.3616011, 0.380962]


