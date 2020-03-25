import torchvision
import torch
from torchvision import transforms
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import random

# 定义 Pytorch 官方给的类别名称，有些是 'N/A' 是已经去掉的类别
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_predection(image_path,threshold):
    img=Image.open(image_path)
    # img=cv2.imread(image_path)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    tf=transforms.ToTensor()
    img=tf(img).to(device)
    pred=model([img])

    pred_bboxes=[[(bbox[0],bbox[1]),(bbox[2],bbox[3])] for bbox in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_scores=pred[0]['scores'].detach().cpu().numpy()
    pred_labels=[COCO_INSTANCE_CATEGORY_NAMES[label] for label in list(pred[0]['labels'].cpu().numpy())]

    pred_scores=pred_scores[pred_scores>threshold]
    pred_bboxes=pred_bboxes[:len(pred_scores)]
    pred_labels=pred_labels[:len(pred_scores)]

    return pred_bboxes,pred_labels

def object_detection(image_path,threshold=0.5):
    pred_bboxes, pred_labels=get_predection(image_path,threshold)
    img=cv2.imread(image_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result_dict={}
    for i in range(len(pred_bboxes)):
        bbox,label=pred_bboxes[i],pred_labels[i]
        color=tuple(random.randint(0,255) for i in range(3))
        cv2.rectangle(
            img,
            bbox[0],
            bbox[1],
            color,
            10
        )
        cv2.putText(
            img,
            label,
            bbox[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
        if label not in result_dict:
            result_dict[label]=1
        else:
            result_dict[label]+=1
    print(result_dict)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    device=torch.device('cpu')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    image_path='./239.jpg'
    threshold = 0.5
    object_detection(image_path,threshold)