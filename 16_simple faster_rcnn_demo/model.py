import torch
from torch import nn

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.features = self._make_layers(cfg)
        self._rpn_model()

        size = (7, 7)
        self.adaptive_max_pool = torch.nn.AdaptiveMaxPool2d(size[0], size[1])
        self.roi_classifier()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _rpn_model(self, mid_channels=512, in_channels=512, n_anchor=9):
        self.rpn_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # I will be going to use softmax here. you can equally use sigmoid if u replace 2 with 1.
        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

        # conv sliding layer
        self.rpn_conv.weight.data.normal_(0, 0.01)
        self.rpn_conv.bias.data.zero_()

        # Regression layer
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()

        # classification layer
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()

    def forward(self, data):
        out_map = self.features(data)
        x = self.rpn_conv(out_map)
        pred_anchor_locs = self.reg_layer(x)  # 回归层，计算有效anchor转为目标框的四个系数
        pred_cls_scores = self.cls_layer(x)  # 分类层，判断该anchor是否可以捕获目标

        return out_map, pred_anchor_locs, pred_cls_scores

    def roi_classifier(self, class_num=20):  # 假设为VOC数据集，共20分类
        # 分类层
        self.roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096),
                                                   nn.ReLU(),
                                                   nn.Linear(4096, 4096),
                                                   nn.ReLU()])
        self.cls_loc = nn.Linear(4096, (class_num+1) * 4)  # (VOC 20 classes + 1 background. Each will have 4 co-ordinates)
        self.cls_loc.weight.data.normal_(0, 0.01)
        self.cls_loc.bias.data.zero_()


        self.score = nn.Linear(4096, class_num+1)  # (VOC 20 classes + 1 background)








