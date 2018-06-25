import numpy as np
import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import default_collate
from minibatch import get_minibatch
import math
from utils import get_max_shape
from encoder import DataEncoder


class COCODataset(data.Dataset):
    """
    A pytorch dataset for coco
    """

    def __init__(self, roidb, training=True):
        self._roidb = roidb
        self.training = training
        self.DATA_SIZE = len(self._roidb)

    def __getitem__(self, index_tuple):
        index, ratio = index_tuple
        single_db = [self._roidb[index]]
        # for one image:
        # blobs: {'data': (ndarray)1 x c x h x w, 'im_info': (ndarray)1 x 3,
        #         'bboxes': (ndarray)1 x num_boxes x 4, 'gt_classes': (ndarray)1 x num_boxes}
        blobs = get_minibatch(single_db)
        # squeeze batch dim
        # blobs: {'data': (ndarray)c x h x w, 'im_info': (ndarray)3,
        #         'bboxes': (ndarray)num_boxes x 4, 'gt_classes': (ndarray)num_boxes}
        for key in blobs:
            blobs[key] = blobs[key].squeeze(axis=0)
        if self._roidb[index]['need_crop']:
            self.crop_data(blobs, ratio)
            # check bounding box
            boxes = blobs['bboxes']
            invalid = (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])
            valid_inds = np.nonzero(~invalid)[0]
            if len(valid_inds) == 0:  # for debug
                print(index, 'index')
                print(self._roidb[index], 'roidb for this index')
                print(boxes, 'boxes')
            if len(valid_inds) < len(boxes):
                for key in ['bboxes', 'gt_classes']:
                    if key in blobs:
                        blobs[key] = blobs[key][valid_inds]
        return blobs

    def crop_data(self, blobs, ratio):
        data_height, data_width = map(int, blobs['im_info'][:2])
        boxes = blobs['bboxes']
        if ratio < 1:  # width << height, crop height
            size_crop = math.ceil(data_width / ratio)  # size after crop
            min_y = math.floor(np.min(boxes[:, 1]))
            max_y = math.floor(np.max(boxes[:, 3]))
            box_region = max_y - min_y
            if min_y == 0:
                y_s = 0
            else:
                if (box_region - size_crop) < 0:
                    y_s_min = max(max_y - size_crop, 0)
                    y_s_max = min(min_y, data_height - size_crop)
                    y_s = y_s_min if y_s_min == y_s_max else np.random.choice(range(int(y_s_min), int(y_s_max) + 1))
                else:
                    # we can't use center crop, because there is a specific situation
                    # for example, boxes: [56, 2, 78, 3], [231, 987, 240, 990]
                    # the box_region is large, when we use center crop, there will be no box.
                    y_s = min_y
            # crop the image
            blobs['data'] = blobs['data'][:, int(y_s):int(y_s + size_crop), :, ]
            # update the im_info
            blobs['im_info'][0] = size_crop
            # shift and clamp boxes ground truth
            boxes[:, 1] -= y_s
            boxes[:, 3] -= y_s
            np.clip(boxes[:, 1], 0, size_crop, out=boxes[:, 1])
            np.clip(boxes[:, 3], 0, size_crop, out=boxes[:, 3])
            blobs['bboxes'] = boxes
        else:  # width >> height, crop width
            size_crop = math.ceil(data_height * ratio)
            min_x = math.floor(np.min(boxes[:, 0]))
            max_x = math.floor(np.max(boxes[:, 2]))
            box_region = max_x - min_x
            if min_x == 0:
                x_s = 0
            else:
                if (box_region - size_crop) < 0:
                    x_s_min = max(max_x - size_crop, 0)
                    x_s_max = min(min_x, data_width - size_crop)
                    x_s = x_s_min if x_s_min == x_s_max else np.random.choice(range(int(x_s_min), int(x_s_max + 1)))
                else:
                    # we can't use center crop, because there is a specific situation
                    # for example, boxes: [2, 56, 3, 78], [987, 231, 990, 240]
                    # the box_region is large, when we use center crop, there will be no box.
                    x_s = min_x
            # crop the image
            blobs['data'] = blobs['data'][:, :, int(x_s):int(x_s + size_crop)]
            # update the im_info
            blobs['im_info'][1] = size_crop
            # shift and clamp boxes ground truth
            boxes[:, 0] -= x_s
            boxes[:, 2] -= x_s
            np.clip(boxes[:, 0], 0, size_crop, out=boxes[:, 0])
            np.clip(boxes[:, 2], 0, size_crop, out=boxes[:, 2])
            blobs['bboxes'] = boxes

    def __len__(self):
        return self.DATA_SIZE


def cal_minibatch_ratio(ratio_list):
    """
    Given the ratio list, we want to make the ratio same for each minibatch on each GPU.
    Note: this only work for 1) MAX_SIZE is ignore during `prep_im_for_blob` and 2) SCALES
    is single scale.
    since all prepared images will have same min side length of scale[0]. we can pad the
    batch image based on that.
    :param ratio_list:
    :return:
            - ratio_list_minibatch: split the ratio_list into minibatches
    """
    DATA_SIZE = len(ratio_list)
    img_per_minibatch = 3
    ratio_list_minibatch = np.empty((DATA_SIZE,))
    num_minibatch = int(np.ceil(DATA_SIZE / img_per_minibatch))
    for i in range(num_minibatch):
        left_idx = i * img_per_minibatch
        right_idx = min((i + 1) * img_per_minibatch - 1, DATA_SIZE - 1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        ratio_list_minibatch[left_idx:(right_idx + 1)] = target_ratio
    return ratio_list_minibatch


class MinibatchSampler(torch_sampler.Sampler):
    def __init__(self, ratio_list, ratio_index):
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.num_data = len(ratio_list)
        self.img_per_minibatch = 3

        self.ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

    def __iter__(self):
        # indices for aspect grouping awared permutation
        n, rem = divmod(self.num_data, self.img_per_minibatch)
        round_num_data = n * self.img_per_minibatch
        indices = np.arange(round_num_data)
        np.random.shuffle(indices.reshape(-1, self.img_per_minibatch))  # inplace shuffle
        # there has a problem, if in each minibatch, the number of image less than 3
        # i think it will raise a error, because the num_pos = 0 for that image or two images
        # and the loss can't backward
        # if rem != 0:
        #     indices = np.append(indices, np.arange(round_num_data, round_num_data + rem))
        ratio_index = self.ratio_index[indices]
        ratio_list_minibatch = self.ratio_list_minibatch[indices]
        return iter(zip(ratio_index.tolist(), ratio_list_minibatch.tolist()))

    def __len__(self):
        return self.num_data


def collate_minibatch(list_of_blobs):
    """
    Stack smaples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence we need to stack samples from each minibatch seperately.
    :param list_of_blobs:
    :return:
    """
    img_per_minibatch = 3
    encoder = DataEncoder()
    Batch = {key: [] for key in ['im_info', 'cls_targets', 'loc_targets', 'data']}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # so we keep roidb in the type of `list of ndarray`.
    # pop method: get the value and delete the key in dicts
    for i in range(0, len(list_of_blobs), img_per_minibatch):
        mini_list = list_of_blobs[i:(i + img_per_minibatch)]
        # pad image data
        mini_list = pad_image_data(mini_list, encoder)
        minibatch = default_collate(mini_list)
        for key in minibatch:
            Batch[key].append(minibatch[key])
    # for Batch, the data format is as follow:
    # Batch = {
    # 'data': [torch.Tensor(3 x 3 x max_h x max_w)] * 8, list
    # 'im_info': [torch.Tensor(3 x 3)] * 8, list
    # 'loc_target': [torch.Tensor(3 x num_loc_target x 4)] * 8, list
    # 'cls_target': [torch.Tensor(3 x num_cls_target)] * 8, list
    # }
    return Batch


def pad_image_data(list_of_blobs, encoder):
    max_shape = get_max_shape([blobs['data'].shape[1:] for blobs in list_of_blobs])
    output_list = []
    for blobs in list_of_blobs:
        data_padded = np.zeros((3, max_shape[0], max_shape[1]), dtype=np.float32)
        _, h, w = blobs['data'].shape
        data_padded[:, :h, :w] = blobs['data']
        blobs['data'] = data_padded
        # get the loc_target and cls_target
        # then the blobs is as follow:
        # for one image
        # blobs = {'data': 3 x max_h x max_w, 'im_info': np.array([h,w,s]),
        # 'loc_targets': torch.Tensor(loc_target), 'cls_targets': torch.Tensor(cls_target)}
        bboxes = blobs['bboxes']
        gt_classes = blobs['gt_classes']
        bboxes = torch.from_numpy(bboxes).float()
        gt_classes = torch.from_numpy(gt_classes).float()
        loc_targets, cls_targets = encoder.encode(bboxes, gt_classes, input_size=(max_shape[1], max_shape[0]))
        blobs['loc_targets'] = loc_targets
        blobs['cls_targets'] = cls_targets
        for k in ['bboxes', 'gt_classes']:
            if k in blobs:
                del blobs[k]
        output_list.append(blobs)
    return output_list
