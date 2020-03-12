import torch
import torchvision
import numpy as np
from model import VGG
import utils
import time

np.random.seed(78)
start=time.time()

bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32) # [y1, x1, y2, x2] format
labels = np.asarray([6, 8], dtype=np.int8)  # 0 represents background
img_tensor = torch.zeros((1, 3, 800, 800)).float()

#******************************** step1 *********************************************

#生成anchors 得到有效anchors及其索引
anchors, valid_anchor_boxes, valid_anchor_index = utils.init_anchor()
#计算有效anchors与bboxs的ious
ious=utils.compute_iou(valid_anchor_boxes, bbox)
#得到正负样本
valid_anchor_len = len(valid_anchor_boxes)
label, argmax_ious = utils.get_pos_neg_sample(ious, valid_anchor_len, pos_iou_threshold=0.7,
                                 neg_iou_threshold=0.3, pos_ratio=0.5, n_sample=256)
#打标gt,得到实际类别与偏移量
max_iou_bbox=bbox[argmax_ious]
anchor_loc=utils.get_coefficient(valid_anchor_boxes, max_iou_bbox)
#得到最终所有anchors的实际类别与偏移量
anchor_label=np.full((len(anchors)),-1,dtype=np.int)
anchor_label[valid_anchor_index]=label

anchor_location=np.full((len(anchors),4),0,dtype=anchor_loc.dtype)
anchor_location[valid_anchor_index,:]=anchor_loc

#******************************** step2 *******************************************
vgg=VGG()
#得到预测类别与偏移量
out_map, pred_anchor_locs, pred_anchor_conf = vgg.forward(img_tensor)

pred_anchor_locs=pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4) #Out: torch.Size([1, 22500, 4])
pred_anchor_conf=pred_anchor_conf.permute(0, 2, 3, 1).contiguous()
objectness_score=pred_anchor_conf.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
pred_anchor_conf=pred_anchor_conf.view(1,-1,2)   # Out torch.size([1, 22500, 2])

# ---------------------step_3: RPN 损失 （有效anchor与预测anchor之间的损失--坐标系数损失与置信度损失）
# 从上面step_1中，我们得到了目标anchor信息：
# 目标anchor坐标系数：anchor_locations  (22500, 4)
# 目标anchor置信度：anchor_conf  (22500,)

# 从上面step_2中，我们得到了预测anchor信息：
# RPN网络预测anchor的坐标系数：pred_anchor_locs  (1, 22500, 4)
# RPN网络预测anchor的置信度: pred_anchor_conf  (1, 22500, 2)

#计算损失时只计算采集的正负样本的loss  忽略掉-1
anchor_location=torch.from_numpy(anchor_location).float() # [22500,4]
anchor_label=torch.from_numpy(anchor_label).long() # [22500,1]
rpn_anchor_locs=pred_anchor_locs[0] #[22500,4]
rpn_anchor_label=pred_anchor_conf[0] #[22500,2]

rpn_loss=utils.rpn_loss(rpn_anchor_locs, rpn_anchor_label, anchor_location, anchor_label, weight=10.0)
print(rpn_loss)

# ---------------------step_4: 根据anchor和预测anchor系数，计算预测框（roi）和预测框的坐标系数(roi_locs)，
# ---------------------并得到每个预测框的所属类别label(roi_labels)
roi, score= utils.get_predict_bbox(anchors, pred_anchor_locs, objectness_score,
                                           n_train_pre_nms=12000, min_size=16)
# 得到的预测框（ROI）还会有大量重叠，再通过NMS（非极大抑制）做进一步的过滤精简
roi = utils.nms(roi, score, nms_thresh=0.7, n_train_post_nms=2000)

#划分正负样本，并得到实际标签与样本的roi
sample_roi, roi_labels, argmax_ious = utils.get_propose_target(roi, bbox, labels,n_sample=128,pos_ratio=0.25,
                                             pos_iou_thresh=0.5,neg_iou_thresh_hi=0.5,neg_iou_thresh_lo=0.0)
#得到样本实际偏移量
roi_locs=utils.get_coefficient(sample_roi,bbox[argmax_ious])

#---------------------------step 5: roi pooling ------------------------------
sample_roi=torch.from_numpy(sample_roi).float()
roi_indices=torch.zeros((len(sample_roi),1))
indices_rois=torch.cat((roi_indices,sample_roi),dim=1).float()

indices_rois[:,1:].mul_(1.0/16.0)
indices_rois=indices_rois.long()  # 取整操作
output=[]
for i in range(len(indices_rois)):
    roi=indices_rois[i]
    indice=int(roi[0])
    feat=out_map.narrow(0,indice,1)[..., roi[1]: (roi[3] + 1), roi[2]: (roi[4] + 1)]
    output.append(vgg.adaptive_max_pool(feat)[0].data)

# ---------------------step_6: Classification 线性分类，预测预测框的类别，置信度和转为目标框的平移缩放系数（要与RPN区分
output=torch.stack(output,dim=0) # [128,512,7,7]
k = output.view(output.size(0), -1)  # [128, 25088]
k = vgg.roi_head_classifier(k)  # (128, 4096)
# torch.Size([128, 84])  84 ==> (20+1)*4,表示每个框有20个候选类别和一个置信度（假设为VOC数据集，共20分类），4表示坐标信息
pred_roi_locs = vgg.cls_loc(k)
# pred_roi_labels： [128, 21] 表示每个框的类别和置信度
pred_roi_labels = vgg.score(k)


# ---------------------step_7: 分类损失  (有效预测框真实系数与有效预测框的预测系数间损失，其中系数是转为目标框的坐标系数)
# 从上面step_4中，我们得到了预测框转为目标框的目标信息：
# 预测框的坐标系数(roi_locs)：  (128, 4)
# 预测框的所属类别(roi_labels)：(128, )

# 从上面step_6中，我们得到了预测框转为目标框的预测信息：
# 预测框的坐标系数：pred_roi_locs  (128, 84)
# 预测框的所属类别和置信度: pred_roi_labels  (128, 21)

roi_locs=torch.from_numpy(roi_locs).float()
roi_labels=torch.from_numpy(roi_labels).long()

pred_roi_locs=pred_roi_locs.reshape(-1,21,4)
pred_roi_locs=pred_roi_locs[torch.arange(len(roi_labels)),roi_labels] #选择真实标签对应的预测回归偏移

roi_loss=utils.roi_loss(pred_roi_locs, pred_roi_labels, roi_locs, roi_labels, weight=10.0)
print(roi_loss)
total_loss=rpn_loss+roi_loss

finish=time.time()
print(finish-start)