from pycocotools.coco import COCO

import os.path
import pickle
import numpy as np
import copy


class COCOJsonDataset(object):
    """
    A class representing a COCO json dataset.
    Load annotation from coco dataset, return the roidb of the COCO, which is a list of dicts.
    """

    def __init__(self, root, annFile, cache_dir):
        """
        :param root: root directory of the COCO dataset
        :param annFile: the name of dataset, with this we can get the annotation file
        :param cache_dir: the directory to save the roidbs
        """
        self.root = root
        self.annFile = annFile
        self.cache_dir = cache_dir
        # the check-up dict of the dataset name with the image directory name
        self._view_map = {
            'minival2014': 'val2014',
            'valminusminival2014': 'val2014',
            'train2014': 'train2014'
        }
        # get the image directory to save the images
        self.img_dir_name = self._view_map[annFile]
        # get the annotation filename
        annFilepath = os.path.join(root, 'annotations/instances_{}.json'.format(annFile))
        # the COCO class
        self._COCO = COCO(annFilepath)
        # get all categories
        cats = self._COCO.loadCats(self._COCO.getCatIds())
        # add the background into categories
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        self.num_classes = len(self._classes)
        # encode the class name to num_code
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        # encode the class name to coco category code
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats], self._COCO.getCatIds()))

    def get_roidb(self):
        """
        Get all the roidbs(dict for each image) in this dataset.
        :return: roidbs(list of dicts)
        """
        # get all the images_ids in this dataset
        img_ids = self._COCO.getImgIds()
        # sort the ids, make each time the same order
        img_ids.sort()
        # load the 'image' of the COCO dataset
        roidb = copy.deepcopy(self._COCO.loadImgs(img_ids))
        for entry in roidb:
            # predefine some attribute of each image
            self._prep_roidb_entry(entry)

        # for cahce_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        cache_file = os.path.join(self.cache_dir, self.annFile + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            self._read_roidb_from_cachefile(roidb, cache_file)
            print('{} gt roidb loaded from {}'.format(self.annFile, cache_file))
        else:
            for entry in roidb:
                self._add_roidb_from_annotations(entry)
            with open(cache_file, 'wb') as f:
                pickle.dump(roidb, f, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))
        return roidb

    def _prep_roidb_entry(self, entry):
        """
        Adds empty metadata fields to an roidb entry
        :param entry: a dict for each image in the dataset
        :return: entry with metadata
        """
        im_path = os.path.join(self.root, 'images', self.img_dir_name, entry['file_name'])
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        # empty placeholders
        entry['bboxes'] = np.empty((0, 4), dtype=np.float32)
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        # remove unwanted fields that come from the json file
        for k in ['date_captured', 'coco_url', 'license', 'url', 'file_name', 'flickr_url']:
            if k in entry:
                del entry[k]

    @property
    def valid_cached_keys(self):
        keys = ['bboxes', 'gt_classes']
        return keys

    def _read_roidb_from_cachefile(self, roidb, cache_file):
        """
        Read gt annotation metadata from cached file
        :param roidb: list of dicts that have added metadata by _prep_roidb_entry
        :param cache_file: the file saves the annotation of roidbs
        :return:
        """
        with open(cache_file, 'rb') as f:
            cached_roidb = pickle.load(f)

        assert len(roidb) == len(cached_roidb)

        for entry, cached_entry in zip(roidb, cached_roidb):
            values = [cached_entry[key] for key in self.valid_cached_keys]
            boxes, gt_classes = values
            entry['bboxes'] = np.append(entry['bboxes'], boxes, axis=0)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)

    def _add_roidb_from_annotations(self, entry):
        """
        Add gt annotation metadata to an roidb entry
        :param entry: a dict for image in the dataset
        :return: entry with annotation
        """
        ann_ids = self._COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self._COCO.loadAnns(ann_ids)
        width = entry['width']
        height = entry['height']
        # valid objs
        # change the annotation boxes from 'xywh' to 'xyxy'
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width, x1 + np.max((0, obj['bbox'][2]))))
            y2 = np.min((height, y1 + np.max((0, obj['bbox'][3]))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_box'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        bboxes = np.zeros((num_objs, 4), dtype=entry['bboxes'].dtype)
        gt_classes = np.zeros((num_objs), dtype=entry['gt_classes'].dtype)

        coco_cat_id_to_class_ind = dict(
            [(self._class_to_coco_cat_id[cls], self._class_to_ind[cls]) for cls in self._classes[1:]])
        for ix, obj in enumerate(objs):
            bboxes[ix, :] = obj['clean_box']
            gt_classes[ix] = coco_cat_id_to_class_ind[obj['category_id']]
        entry['bboxes'] = np.append(entry['bboxes'], bboxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
