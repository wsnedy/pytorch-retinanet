import os, sys

lib_path_datasets = os.path.abspath(os.path.join('..', 'datasets'))
sys.path.append(lib_path_datasets)
from encoder import DataEncoder
from minibatch import _get_image_blob
from pycocotools.cocoeval import COCOeval
import torch
import torch.utils.data
from torch.autograd import Variable
import json


def evaluate_coco(eval_dataset, net):
    """
    The function is to evaluate the network on coco dataset.
    :param eval_dataset: the COCOJsonDataset object for the evaluation.
    :param net: the model.
    :return: 
            - coco_eval.stats: the AP of the model on coco dataset.
    """
    encoder = DataEncoder()
    # for roidbs
    eval_roidb = eval_dataset.get_roidb()
    # for net, change the mode to eval
    net.eval()
    # get the dict of class_ind to coco_cat_id
    class_ind_to_coco_cat_id = dict(
        [(eval_dataset._class_to_ind[cls], eval_dataset._class_to_coco_cat_id[cls]) for cls in
         eval_dataset._classes[1:]])
    results = []
    for idx, entry in enumerate(eval_roidb):
        roidb = [entry]
        blob, im_scale = _get_image_blob(roidb)
        h, w = blob.shape[2:]
        scale = im_scale[0]
        img = torch.from_numpy(blob).float()
        img = [Variable(img)]
        loc_targets, cls_targets = [Variable(torch.zeros(1, 4))], [Variable(torch.zeros(1))]
        loc_preds, cls_preds = net(img, loc_targets, cls_targets)
        try:
            boxes, labels, scores = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
        except:
            continue
        # rescale the boxes to original image size
        boxes = boxes / torch.Tensor([scale, scale, scale, scale]).cuda()
        boxes = torch.cat([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)
        for i in range(len(labels)):
            img_result = {
                'image_id': entry['id'],
                'category_id': class_ind_to_coco_cat_id[int(labels[i])],
                'score': float(scores[i]),
                'bbox': boxes[i].tolist()
            }
            results.append(img_result)
    # write output
    json.dump(results, open('../results/detection_result.json', 'w'), indent=4)
    # laod result in coco eval tool
    coco_true = eval_dataset._COCO
    coco_pred = coco_true.loadRes('../results/detection_result.json')
    image_ids = coco_true.getImgIds()
    # run coco evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats
