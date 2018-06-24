import os
import sys

sys.path.append(os.path.join('..'))
import torch
import torch.nn as nn

from .fpn import FPN50
from torch.autograd import Variable
from lib.loss import FocalLoss


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=80):
        super(RetinaNet, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
        self.focal_loss = FocalLoss()

    def forward(self, x, loc_targets, cls_targets):
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
        prediction['loc_preds'] = loc_preds.view(-1, 4)
        prediction['cls_preds'] = cls_preds.view(-1, 80)
        num_pos = loc_loss = cls_loss = 0
        if self.training:
            num_pos, loc_loss, cls_loss = self.focal_loss(loc_preds, loc_targets, cls_preds, cls_targets)
        # return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)
        return prediction, num_pos, loc_loss, cls_loss

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


def test():
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(2, 3, 224, 224)))
    print(loc_preds.size())
    print(cls_preds.size())
    loc_grads = Variable(torch.randn(loc_preds.size()))
    cls_grads = Variable(torch.randn(cls_preds.size()))
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)

    # test()
