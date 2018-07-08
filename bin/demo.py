import os, sys

lib_path_datasets = os.path.abspath(os.path.join('..', 'datasets'))
lib_path_network = os.path.abspath(os.path.join('..', 'network'))
sys.path.append(lib_path_datasets)
sys.path.append(lib_path_network)
from encoder import DataEncoder
from coco_json_dataset import COCOJsonDataset
from minibatch import _get_image_blob
from retinanet import RetinaNet
from torch.autograd import Variable
import torch
import cv2
import numpy as np
import time

# network
net = RetinaNet()
net.load_state_dict(torch.load('../pretrained_model/net.pth'))
print('==> Resuming from checkpoint..')
checkpoint = torch.load('../checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

# encoder
encoder = DataEncoder()

# inputs
# get the dict of class_ind to coco_cat_id
root = '/mnt/xfs1/home/chenqiang/data/coco'
cache_dir = '../cache_dir'
eval_dataset_name = 'minival2014'
eval_dataset = COCOJsonDataset(root=root, annFile=eval_dataset_name, cache_dir=cache_dir)
class_ind_to_category = dict(
    [(eval_dataset._class_to_ind[cls], cls) for cls in eval_dataset._classes[1:]])

# read the image
roidb = {}
image_file = '../images/sample.jpg'
roidb['image'] = image_file
blob, scale = _get_image_blob([roidb])
h, w = blob.shape[2:]
scale = scale[0]
img = torch.from_numpy(blob).float()
img = [Variable(img, volatile=True)]
loc_targets, cls_targets = [Variable(torch.zeros(1, 4))], [Variable(torch.zeros(1))]
loc_preds, cls_preds = net(img, loc_targets, cls_targets)
boxes, labels, scores = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
# rescale the boxes to original image size
boxes = boxes / scale
boxes = torch.cat([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)


# draw
def vis_detections(im, dets, labels, scores, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(dets.shape[0]):
        bbox = tuple(int(np.round(x)) for x in dets[i])
        class_name = class_ind_to_category[labels[i]]
        score = scores[i]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im
# read image
im = np.array(cv2.imread(image_file))
# change the BGR->RGB
im = im[:, :, ::-1]
im2show = vis_detections(im, boxes.cpu().numpy(), labels, scores)
result_path = os.path.join(image_file.split('.jpg')[0] + "_det.jpg")
cv2.imwrite(result_path, im2show)