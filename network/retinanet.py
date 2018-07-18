import os, sys
lib_path = os.path.abspath(os.path.join('..', 'loss'))
sys.path.append(lib_path)
import torch
import torch.nn as nn

from fpn import FPN50
from torch.autograd import Variable
from focal_loss import FocalLoss


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=80):
        super(RetinaNet, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        self.focal_loss = FocalLoss()

    def forward(self, inputs):
        x, loc_targets, cls_targets = inputs
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        prediction = {}
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            # [N, 9*4, H, W] -> [N, H, W, 9*4] -> [N, H*W*9, 4]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            # [N, 9*80, H, W] -> [N, H, W, 9*80] -> [N, H*W*9, 80]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        if self.training:
            loc_loss, cls_loss = self.focal_loss(loc_preds, loc_targets, cls_preds, cls_targets)
            return loc_loss, cls_loss
        # return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)
        else:
            return loc_preds, cls_preds

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        """
        Freeze BatchNorm layers.
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

