import numpy as np
import cv2
from utils import prep_im_for_blob, im_list_to_blob


def get_minibatch(roidb):
    """
    Given a roidb, construct a minibatch sampled from it.
    :param roidb: length is 1, just [single_db]
    :return:
            - blobs: a new dict for each single_db
    """
    # we collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blob_names = ['data', 'im_info', 'bboxes', 'gt_classes']
    blobs = {k: [] for k in blob_names}
    # get the input image blob
    im_blob, im_scales = _get_image_blob(roidb)
    blobs['data'] = im_blob
    # get the im_info, bboxes and gt_classes
    _get_roidb_im_info(blobs, im_scales, roidb)
    # for blobs: {'data': (ndarray)1 x c x h x w, 'im_info': (ndarray)1 x 3,
    #             'bboxes': (ndarray)1 x num_boxes x 4, 'gt_classes': (ndarray)1 x num_boxes}
    return blobs


def _get_image_blob(roidb):
    """
    Builds an input blob from the images in the roidb at the specified scales.
    :param roidb:
    :return:
    """
    num_images = len(roidb)
    IMAGE_SIZE = (600,)
    MAX_SIZE = 1000
    PIXEL_MEAN = np.array([[[102.9801, 115.9465, 122.7717]]])
    # sample random image size to use for each image in this batch
    # size_inds is np.array([ind])
    size_inds = np.random.randint(0, high=len(IMAGE_SIZE), size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, 'Failed to read image \'{}\''.format(roidb[i]['image'])
        # if flipped, flip the image
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = IMAGE_SIZE[size_inds[i]]
        im, im_scale = prep_im_for_blob(im, PIXEL_MEAN, [target_size], MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])
    # create a blob to hold the input images [n, c, h, w]
    blob = im_list_to_blob(processed_ims)
    return blob, im_scales


def _get_roidb_im_info(blobs, im_scales, roidb):
    valid_keys = ['bboxes', 'gt_classes']
    for im_i, entry in enumerate(roidb):
        scale = im_scales[im_i]
        im_height = np.round(entry['height'] * scale)
        im_width = np.round(entry['width'] * scale)
        im_info = np.array([[im_height, im_width, scale]], dtype=np.float32)
        blobs['im_info'].append(im_info)
        # for roidb
        for k in valid_keys:
            if k in entry:
                # rescale the boxes with scale for image
                if k == 'bboxes':
                    blobs[k].append(entry[k] * scale)
                else:
                    blobs[k].append(entry[k])
    blobs['im_info'] = np.array(blobs['im_info']).squeeze(axis=0)
    blobs['bboxes'] = np.array(blobs['bboxes'])
    blobs['gt_classes'] = np.array(blobs['gt_classes'])
