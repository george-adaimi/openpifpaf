from collections import defaultdict
import copy
import logging
import os
from .constants import COCO_CATEGORIES

import numpy as np
import torch.utils.data
from PIL import Image

from .. import transforms, utils


LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))

class PIF_Category(object):
    def __init__(self, num_classes, catID_label):
        self.num_classes = num_classes
        self.catID_label = catID_label

    def __call__(self, anns):
        for ann in anns:
            temp_id = self.catID_label[ann['category_id']]
            bbox = ann['bbox']
            temp = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00]*(temp_id) + [bbox[0], bbox[1], 2 , bbox[0]+ bbox[2]/2, bbox[1] + bbox[3]/2 , 2] + [0.00, 0.00, 0.00, 0.00, 0.00, 0.00]*(self.num_classes-(temp_id+1))
            ann['keypoints'] = copy.deepcopy(temp)
            ann['num_keypoints'] = 2
        return anns

class Coco(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        image_dir (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
    """

    categories = np.asarray(COCO_CATEGORIES)[[0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37,38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 69, 71, 72, 73, 74, 75, 76,77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89]]
    def __init__(self, image_dir, ann_file, *, target_transforms=None,
                 n_images=None, preprocess=None,
                 category_ids=None,
                 image_filter='keypoint-annotations'):
        if category_ids is None:
            category_ids = [1]

        from pycocotools.coco import COCO  # pylint: disable=import-outside-toplevel
        self.image_dir = image_dir
        self.coco = COCO(ann_file)

        self.category_ids = category_ids

        if image_filter == 'all':
            self.ids = self.coco.getImgIds()
        elif image_filter == 'annotated':
            self.ids = self.coco.getImgIds(catIds=self.category_ids)
            self.filter_for_annotations()
        elif image_filter == 'keypoint-annotations':
            self.ids = self.coco.getImgIds(catIds=self.category_ids)
            self.filter_for_keypoint_annotations()
        else:
            raise Exception('unknown value for image_filter: {}'.format(image_filter))

        if n_images:
            self.ids = self.ids[:n_images]
        LOG.info('Images: %d', len(self.ids))

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

        # Cat ID (missing class)
        self.cat_ids = self.coco.getCatIds()
        self.catID_label = {catid:label for label, catid in enumerate(self.cat_ids)}
        self.PIF_category = PIF_Category(num_classes=len(self.cat_ids), catID_label=self.catID_label)

    def filter_for_keypoint_annotations(self):
        LOG.info('filter for keypoint annotations ...')
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        LOG.info('... done.')

    def filter_for_annotations(self):
        """removes images that only contain crowd annotations"""
        LOG.info('filter for annotations ...')
        def has_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if ann.get('iscrowd'):
                    continue
                return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_annotation(image_id)]
        LOG.info('... done.')

    def class_aware_sample_weights(self, max_multiple=10.0):
        """Class aware sampling.

        To be used with PyTorch's WeightedRandomSampler.

        Reference: Solution for Large-Scale Hierarchical Object Detection
        Datasets with Incomplete Annotation and Data Imbalance
        Yuan Gao, Xingyuan Bu, Yang Hu, Hui Shen, Ti Bai, Xubin Li and Shilei Wen
        """
        ann_ids = self.coco.getAnnIds(imgIds=self.ids, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)

        category_image_counts = defaultdict(int)
        image_categories = defaultdict(set)
        for ann in anns:
            if ann['iscrowd']:
                continue
            image = ann['image_id']
            category = ann['category_id']
            if category in image_categories[image]:
                continue
            image_categories[image].add(category)
            category_image_counts[category] += 1

        weights = [
            sum(
                1.0 / category_image_counts[category_id]
                for category_id in image_categories[image_id]
            )
            for image_id in self.ids
        ]
        min_w = min(weights)
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))
        max_w = min_w * max_multiple
        weights = [min(w, max_w) for w in weights]
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))

        return weights

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]
        LOG.debug(image_info)
        with open(os.path.join(self.image_dir, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # bbox center annotation
        anns = self.PIF_category(anns)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, meta)

        # mask valid TODO still necessary?
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        LOG.debug(meta)

        # log stats
        for ann in anns:
            if getattr(ann, 'iscrowd', False):
                continue
            if not np.any(ann['keypoints'][:, 2] > 0.0):
                continue
            STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})

        # transform targets
        if self.target_transforms is not None:
            anns = [t(image, anns, meta) for t in self.target_transforms]

        return image, anns, meta

    def __len__(self):
        return len(self.ids)
