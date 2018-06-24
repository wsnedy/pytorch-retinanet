import numpy as np
from coco_json_dataset import COCOJsonDataset


def combined_roidb_for_training(root, dataset_names, cache_dir):
    """
    Load and concatenate roidbs for one or more datasets, the roidb entries are
    then prepared for use in training, which involves caching certain types of
    metadata for each roidb entry.
    :param root: root directory of the COCO dataset
    :param dataset_names: list of dataset_name, with this we can get each the annotation file
    :param cache_dir: the directory to save the roidbs
    :return: 
            - roidb: list of dicts which contain the metadata in each entry
            - ratio_list: the list if ratio for each image in the dataset
            - ratio_index: the ranked index for dataset
    """

    def get_roidb(root, dataset_name, cache_dir):
        dataset = COCOJsonDataset(root=root, annFile=dataset_name, cache_dir=cache_dir)
        roidb = dataset.get_roidb()
        extend_with_flipped_entries(roidb)
        return roidb

    roidbs = [get_roidb(root, dataset_name, cache_dir) for dataset_name in dataset_names]
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    roidb = filter_for_training(roidb)

    print('Computing image aspect ratios and ordering the ratios....')
    ratio_list, ratio_index = rank_for_training(roidb)
    print('Done!!!')

    return roidb, ratio_list, ratio_index


def extend_with_flipped_entries(roidb):
    """
    Flip each entry in the given roidb and return a new roidb that is the
    concatenation of the original roidb and the flipped entries.

    "Flipping" an entry means that that image and associated metadata(
    gt boxes) are horizontally flipped
    :param roidb: list of dicts which contain the metadata in each entry
    :return:
            - roidb: concatenation of the original roidb and flipped roidb
    """
    flipped_roidb = []
    for entry in roidb:
        width = entry['width']
        boxes = entry['bboxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2
        boxes[:, 2] = width - oldx1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        flipped_entry = {}
        dont_copy = ('bboxes', 'flipped')
        for k, v in entry.items():
            if k not in dont_copy:
                flipped_entry[k] = v
        flipped_entry['bboxes'] = boxes
        flipped_entry['flipped'] = True
        flipped_roidb.append(flipped_entry)
    roidb.extend(flipped_roidb)


def filter_for_training(roidb):
    """
    Remove roidb entries that have no gt_boxes.
    :param roidb: list of dicts which contain the metadata in each entry
    :return:
            -roidb: roidb with empty annotation removed
    """

    def is_valid(entry):
        valid = len(entry['bboxes']) > 0
        if valid:
            assert len(entry['gt_classes']) > 0, 'gt_classes is not valid'
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after, num, num_after))
    return filtered_roidb


def rank_for_training(roidb):
    """
    Rank the roidb entries according to image aspect ratio and mark for cropping
    for efficient batching if image is too long.
    :param roidb:
    :return:
            ratio_list: ndarray, list of aspect ratios from small to large
            ratio_index: ndarray, list of roidb entry indices correspond to the ratios
    """
    RATIO_HI = 2  # largest ratio to preserve
    RATIO_LO = 0.5  # smallest ratio to preserve
    need_crop_cnt = 0

    ratio_list = []
    for entry in roidb:
        width = entry['width']
        height = entry['height']
        ratio = width / float(height)

        if ratio > RATIO_HI:
            entry['need_crop'] = True
            ratio = RATIO_HI
            need_crop_cnt += 1
        elif ratio < RATIO_LO:
            entry['need_crop'] = True
            ratio = RATIO_LO
            need_crop_cnt += 1
        else:
            entry['need_crop'] = False
        ratio_list.append(ratio)

    print('Number of entries that need to be cropped:'
          ' {}. Ratio bound: [{}, {}]'.format(need_crop_cnt, RATIO_LO, RATIO_HI))

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)

    return ratio_list[ratio_index], ratio_index
