import numpy as np
import cv2
import torch


def prep_im_for_blob(im, pixel_means, target_sizes, max_size):
    """
    Prepare an image for use as a network input blob, specially:
    - Subtract per-channel pixel mean
    - Convert to float32
    - Rescale to each of the specified target size (capped at max_size)
    :param im: the image ndarray
    :param pixel_means: image means for each channel
    :param target_sizes: the target size for rescale the image
    :param max_size: the max size of the longer side in the image
    :return:
            - A list of transformed images, one for each target size. Also returns the scale
            factors that were used to compute each returned image.
    """
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    ims = []
    im_scales = []
    for target_size in target_sizes:
        im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
        im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        ims.append(im_resized)
        im_scales.append(im_scale)
    return ims, im_scales


def get_target_scale(im_size_min, im_size_max, target_size, max_size):
    """
    Calculate target resize scale
    :param im_size_min: the shorter side of image
    :param im_size_max: the longer side of image
    :param target_size: target size
    :param max_size: max size for the longer side of image
    :return:
            - im_scale: the scale for rescale the image
    """
    im_scale = float(target_size) / float(im_size_min)
    # prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale


def im_list_to_blob(ims):
    """
    Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
        - BGR channel order
        - pixel means substracted
        - resized to the desired input size
        - float32 numpy ndarray format
    :param ims:
    :return:
            - a 4D NCHW tensor of the images concatenated along axis 0 with shape.
    """
    if not isinstance(ims, list):
        ims = [ims]
    max_shape = get_max_shape([im.shape[:2] for im in ims])
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # move channels (axis 3) to (axis 1)
    # axis order will become: (batch, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def get_max_shape(im_shapes):
    """
    Calculate max spatial size (h, w) for batching given a list of image shapes
    :param im_shapes: list of np.array([h, w])
    :return: 
            - max_shape: the max_shape in the list of images
    """
    max_shape = np.array(im_shapes).max(axis=0)
    assert max_shape.size == 2
    # pad the image so they can be divisible by a stride
    stride = 32
    max_shape[0] = int(np.ceil(max_shape[0] / float(stride)) * stride)
    max_shape[1] = int(np.ceil(max_shape[1] / float(stride)) * stride)
    return max_shape


def meshgrid(x, y, row_major=True):
    """
    Return meshgrid in range x & y
    :param x: (int) first dim range
    :param y: (int) second dim range
    :param row_major: (bool) row major or column major.
    :return: (tensor) meshgrid, sized [x*y, 2]

    Example:
    >> meshgrid(3, 2)
    0 0
    1 0
    2 0
    0 1
    1 1
    2 1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3, 2, row_major=False)
    0 0
    0 1
    0 2
    1 0
    1 1
    1 2
    [torch.FloatTensor of size 6x2]
    """
    a = torch.arange(0, x)
    b = torch.arange(0, y)
    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
    return torch.cat([xx, yy], 1) if row_major else torch.cat([yy, xx], 1)


def change_box_order(boxes, order):
    """
    Change box order between (xmin, ymin, xmax, ymax) and (xcenter, ycenter, width, height).
    :param boxes: (tensor) bounding boxes, sized [N, 4]
    :param order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.
    :return: (tensor) converted bounding boxes, size [N, 4]
    """
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a + b) / 2, b - a], 1)
    return torch.cat([a - b / 2, a + b / 2], 1)


def box_iou(box1, box2, order='xyxy'):
    """
    Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).
    :param box1: (tensor) bounding boxes, sized [N, 4]
    :param box2: (tensor) bounding boxes, sized [M, 4].
    :param order: (str) box order, either 'xyxy' or 'xywh'
    :return: (tensor) iou, sized [N, M]
    """
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [N.]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    """
    Non Maximum suppression.
    :param bboxes: (tensor) bounding boxes, size [N, 4]
    :param scores: (tensor) bbox scores, sized [N,].
    :param threshold: (float) overlap threshold
    :param mode: (str) 'union' or 'min'
    :return:
        keep: (tensor) selected indices.
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    # import pdb; pdb.set_trace()

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr < threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        # because the length of the ovr is less than the order by 1
        # so we have to add to ids to get the right one
        order = order[ids + 1]
    return torch.LongTensor(keep)


def one_hot_embedding(labels, num_classes):
    """
    Embedding labels to one-hot form.

    Args:
    :param labels: (LongTensor) class label, sized [N,].
    :param num_classes: (int) number of classes.
    :return:
            (tensor) encoded labels, size [N, #classes].
    """
    y = torch.eye(num_classes)  # [D, D]
    return y[labels]  # [N, D]
