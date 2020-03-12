import torch
import numpy as np
from torch.nn import functional as F

def init_anchor(img_size=800, sub_sample=16):
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]  # 该尺寸是针对特征图的
    feature_size = (img_size // sub_sample)

    mesh_scales, mesh_ratios = np.meshgrid(anchor_scales, ratios)
    mesh_h = sub_sample * mesh_scales * np.sqrt(mesh_ratios)
    mesh_w = sub_sample * mesh_scales * np.sqrt(1.0 / mesh_ratios)
    mesh_h = mesh_h.reshape(1, -1).repeat(feature_size * feature_size, axis=0).reshape(-1, 1)
    mesh_w = mesh_w.reshape(1, -1).repeat(feature_size * feature_size, axis=0).reshape(-1, 1)

    ctr_x = np.arange(sub_sample, (feature_size + 1) * sub_sample, sub_sample)  # 共feature_size个
    ctr_y = np.arange(sub_sample, (feature_size + 1) * sub_sample, sub_sample)  # 共feature_size个
    ctr_y,ctr_x=np.meshgrid(ctr_y,ctr_x)
    ctr = np.concatenate((ctr_y.reshape(-1,1), ctr_x.reshape(-1,1)), axis=1) - 8

    ctrs = np.repeat(ctr, 9, axis=0)
    anchors = np.zeros(((feature_size * feature_size * 9), 4))
    anchors[:, 0::4] = ctrs[:, 0::2] - mesh_h / 2.0
    anchors[:, 1::4] = ctrs[:, 1::2] - mesh_w / 2.0
    anchors[:, 2::4] = ctrs[:, 0::2] + mesh_h / 2.0
    anchors[:, 3::4] = ctrs[:, 1::2] + mesh_w / 2.0

    valid_anchor_index = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= 800) &
        (anchors[:, 3] <= 800)
    )[0]  # 该函数返回数组中满足条件的index
    # print valid_anchor_index.shape  # (8940,)，表明有8940个框满足条件

    # 获取有效anchor（即边框都在图片内的anchor）的坐标
    valid_anchor_boxes = anchors[valid_anchor_index]
    return anchors, valid_anchor_boxes, valid_anchor_index

def compute_iou(valid_anchor_boxes, bbox):
    ious=np.zeros((len(valid_anchor_boxes),len(bbox)))
    for i,anchor_box in enumerate(valid_anchor_boxes):
        anchor_y1,anchor_x1,anchor_y2,anchor_x2=anchor_box
        anchor_area=(anchor_y2-anchor_y1+1)*(anchor_x2-anchor_x1+1)
        areas=(bbox[:,3]-bbox[:,1]+1)*(bbox[:,2]-bbox[:,0]+1)

        x1_max=np.maximum(anchor_x1,bbox[:,1])
        y1_max=np.maximum(anchor_y1,bbox[:,0])
        x2_min=np.minimum(anchor_x2,bbox[:,3])
        y2_min=np.minimum(anchor_y2,bbox[:,2])
        w=np.maximum(0,x2_min-x1_max+1)
        h=np.maximum(0,y2_min-y1_max+1)
        union_area=w*h
        ious[i,:]=union_area/(areas+anchor_area-union_area)
    return ious

def get_pos_neg_sample(ious, valid_anchor_len, pos_iou_threshold=0.7,neg_iou_threshold=0.3, pos_ratio=0.5, n_sample=256):
    gt_max_ious=np.max(ious,axis=0)
    gt_argmax_ious=np.where(ious==gt_max_ious)[0]

    max_ious=np.max(ious,axis=1)
    argmax_ious=np.argmax(ious,axis=1)

    label=np.full((valid_anchor_len),-1,dtype=np.int)
    label[max_ious<neg_iou_threshold]=0
    label[gt_argmax_ious]=1     #正样本 case1
    label[max_ious>=pos_iou_threshold]=1

    n_pos=n_sample*pos_ratio
    pos_index=np.where(label==1)[0]
    if len(pos_index)>n_pos:
        disable_pos=np.random.choice(pos_index,(len(pos_index)-n_pos),replace=False)
        label[disable_pos]=-1

    n_neg=n_sample-np.sum(label==1)
    neg_index=np.where(label==0)[0]
    if len(neg_index)>n_neg:
        disable_neg=np.random.choice(neg_index,(len(neg_index)-n_neg),replace=False)
        label[disable_neg]=-1
    return label,argmax_ious

def get_coefficient(valid_anchor_boxes, max_iou_bbox):
    anchor_w=valid_anchor_boxes[:,3]-valid_anchor_boxes[:,1]
    anchor_h=valid_anchor_boxes[:,2]-valid_anchor_boxes[:,0]
    anchor_ctr_x = valid_anchor_boxes[:,1]+0.5*anchor_w
    anchor_ctr_y=valid_anchor_boxes[:,0]+0.5*anchor_h

    bbox_w=max_iou_bbox[:,3]-max_iou_bbox[:,1]
    bbox_h=max_iou_bbox[:,2]-max_iou_bbox[:,0]
    bbox_ctr_x=max_iou_bbox[:,1]+0.5*bbox_w
    bbox_ctr_y=max_iou_bbox[:,0]+0.5*bbox_h

    eps = np.finfo(anchor_h.dtype).eps
    anchor_h = np.maximum(anchor_h, eps)
    anchor_w = np.maximum(anchor_w, eps)

    tx=(bbox_ctr_x-anchor_ctr_x)/anchor_w
    ty=(bbox_ctr_y-anchor_ctr_y)/anchor_h
    tw=np.log(bbox_w/anchor_w)
    th=np.log(bbox_h/anchor_h)

    anchor_loc=np.vstack((ty,tx,th,tw)).transpose()
    return anchor_loc


def rpn_loss(rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_label, weight=10.0):
    #分类损失
    rpn_cls_loss=F.cross_entropy(rpn_score,gt_rpn_label,ignore_index=-1)

    #回归损失
    pos_index=torch.where(gt_rpn_label>0)[0]
    pos_loc=rpn_loc[pos_index]
    pos_gt_loc=gt_rpn_loc[pos_index]

    x=torch.abs(pos_loc.data-pos_gt_loc.data)
    rpn_loc_loss=(x<1).float() * 0.5 * x**2 + (x>=1).float() * (x-0.5)
    rpn_loc_loss=torch.sum(rpn_loc_loss,dim=1).mean()

    #总loss
    rpn_loss=rpn_cls_loss+weight*rpn_loc_loss
    return rpn_loss

def get_predict_bbox(anchors, pred_anchor_locs, objectness_score,n_train_pre_nms=12000, min_size=16):
    pred_anchor_locs_numpy=pred_anchor_locs[0].data.numpy()
    objectness_score_numpy=objectness_score[0].data.numpy()

    anchor_w=anchors[:,3]-anchors[:,1]
    anchor_h=anchors[:,2]-anchors[:,0]
    anchor_ctr_x = anchors[:,1]+0.5*anchor_w
    anchor_ctr_y=anchors[:,0]+0.5*anchor_h

    ty = pred_anchor_locs_numpy[:, 0]
    tx=pred_anchor_locs_numpy[:,1]
    th=pred_anchor_locs_numpy[:,2]
    tw=pred_anchor_locs_numpy[:,3]

    pred_anchor_ctr_x=tx * anchor_w + anchor_ctr_x
    pred_anchor_ctr_y=ty * anchor_h + anchor_ctr_y
    pred_anchor_w=np.exp(tw) * anchor_w
    pred_anchor_h=np.exp(th) * anchor_h

    #初步得到rois
    rois=np.zeros((len(anchors),4),dtype=pred_anchor_locs_numpy.dtype)
    rois[:,0]=pred_anchor_ctr_y - 0.5 * pred_anchor_h
    rois[:,1]=pred_anchor_ctr_x - 0.5 * pred_anchor_w
    rois[:,2]=pred_anchor_ctr_y + 0.5 * pred_anchor_h
    rois[:,3]=pred_anchor_ctr_x + 0.5 * pred_anchor_w

    #修剪
    img_size = (800, 800)
    rois[:,[0,2]]=np.clip(rois[:,[0,2]],0,img_size[0])
    rois[:, [1, 3]] = np.clip(rois[:, [1, 3]], 0, img_size[1])

    #去除不满足宽高限制的框
    hs = rois[:, 2] - rois[:, 0]
    ws = rois[:, 3] - rois[:, 1]
    keep=np.where((hs>=min_size) & (ws>=min_size))[0]
    rois=rois[keep]
    score=objectness_score_numpy[keep]

    #初步帅选12000
    order=score.ravel().argsort()[::-1]
    order=order[:n_train_pre_nms]
    score=score[order]
    rois=rois[order]
    return rois,score

def nms(roi, score, nms_thresh=0.7, n_train_post_nms=2000):
    order=score.argsort()[::-1]
    y1=roi[:,0]
    x1=roi[:,1]
    y2=roi[:,2]
    x2=roi[:,3]
    areas=(y2-y1+1)*(x2-x1+1)

    keep=[]
    while order.size>0:
        i=order[0]
        keep.append(i)

        y1_max=np.maximum(y1[i],y1[order[1:]])
        x1_max=np.maximum(x1[i],x1[order[1:]])
        y2_min=np.minimum(y2[i],y2[order[1:]])
        x2_min=np.minimum(x2[i],x2[order[1:]])
        w=np.maximum(0,x2_min-x1_max+1)
        h=np.maximum(0,y2_min-y1_max+1)
        union_area=w*h

        ious=union_area/(areas[i]+areas[order[1:]]-union_area)
        keep_inds=np.where(ious<=nms_thresh)[0]
        order=order[keep_inds+1]
    keep=keep[:n_train_post_nms]
    roi=roi[keep]
    return roi

def get_propose_target(roi, bbox, labels,n_sample=128,pos_ratio=0.25,
                       pos_iou_thresh=0.5,neg_iou_thresh_hi=0.5,neg_iou_thresh_lo=0.0):
    ious=compute_iou(roi,bbox)
    argmax_ious=np.argmax(ious,axis=1)
    max_ious=np.max(ious,axis=1)
    gt_argmax_label=labels[argmax_ious]

    #正样本
    n_pos=n_sample*pos_ratio
    pos_index=np.where(max_ious>=pos_iou_thresh)[0]
    num_pos=int(min(len(pos_index),n_pos))
    if pos_index.size>0:
        pos_index=np.random.choice(pos_index,num_pos,replace=False)

    #负样本
    n_neg=n_sample-len(pos_index)
    neg_index=np.where((max_ious>=neg_iou_thresh_lo) & (max_ious<neg_iou_thresh_hi))[0]
    num_neg=int(min(len(neg_index),n_neg))
    if neg_index.size>0:
        neg_index=np.random.choice(neg_index,num_neg,replace=False)

    keep_index=np.append(pos_index,neg_index)
    gt_argmax_label=gt_argmax_label[keep_index]
    gt_argmax_label[num_pos:]=0 #所有负样本标签设为0
    sample_roi=roi[keep_index]
    argmax_ious=argmax_ious[keep_index]
    return sample_roi,gt_argmax_label,argmax_ious


def roi_loss(roi_loc, pred_roi_labels, gt_roi_loc, gt_roi_label, weight=10.0):
    #分类损失
    roi_cls_loss=F.cross_entropy(pred_roi_labels,gt_roi_label,ignore_index=-1)

    #回归损失
    pos_index=torch.where(gt_roi_label>0)[0]
    pos_loc=roi_loc[pos_index]
    pos_gt_loc=gt_roi_loc[pos_index]

    x=torch.abs(pos_loc.data-pos_gt_loc.data)
    roi_reg_loss=(x<1).float() * 0.5 * x**2 + (x>=1).float() * (x - 0.5)
    roi_reg_loss=torch.sum(roi_reg_loss,dim=1).mean()

    #总损失
    roi_loss=roi_cls_loss+weight*roi_reg_loss
    return roi_loss





