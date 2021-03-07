from collections import defaultdict
import copy
import logging
import os
import numpy as np

import torch.utils.data
from PIL import Image

from openpifpaf import transforms, utils


LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class VG(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        image_dir (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
    """

    def __init__(self, image_dir, ann_file, *,
                 n_images=None, preprocess=None,
                 category_ids=None):
        if category_ids is None:
            category_ids = []

        from pycocotools.coco import COCO  # pylint: disable=import-outside-toplevel
        self.image_dir = image_dir
        self.coco = COCO(ann_file)

        self.ids = list(self.coco.dataset.keys())
        if n_images:
            self.ids = self.ids[:n_images]
        LOG.info('Images: %d', len(self.ids))

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM


    def get_frequency_prior(self, obj_categories, rel_categories):
        fg_matrix = np.zeros((
            len(obj_categories),  # not include background
            len(obj_categories),  # not include background
            len(rel_categories),  # include background
        ), dtype=np.int64)

        bg_matrix = np.zeros((
            len(obj_categories),  # not include background
            len(obj_categories),  # not include background
        ), dtype=np.int64)

        smoothing_pred = np.zeros(len(rel_categories), dtype=np.float32)

        count_pred = 0.0
        for image_id in self.ids:
            # get all object boxes
            gt_box_to_label = {}
            targets = self.coco.dataset[image_id]
            for i, target in enumerate(targets):
                prd_lbl = target['predicate']
                x = target['subject']['bbox'][2]
                y = target['subject']['bbox'][0]
                w = target['subject']['bbox'][3] - x
                h = target['subject']['bbox'][1] - y
                sbj_lbl = int(target['subject']['category'])

                sbj_box = [x,y,w,h]
                x1 = target['object']['bbox'][2]
                y1 = target['object']['bbox'][0]
                w1 = target['object']['bbox'][3] - x1
                h1 = target['object']['bbox'][1] - y1
                obj_lbl = int(target['object']['category'])
                obj_box = [x1,y1,w1,h1]

                if tuple(sbj_box) not in gt_box_to_label:
                    gt_box_to_label[tuple(sbj_box)] = sbj_lbl
                if tuple(obj_box) not in gt_box_to_label:
                    gt_box_to_label[tuple(obj_box)] = obj_lbl


                fg_matrix[sbj_lbl, obj_lbl, prd_lbl] += 1

                for b1, l1 in gt_box_to_label.items():
                    for b2, l2 in gt_box_to_label.items():
                        if b1 == b2:
                            continue
                        bg_matrix[l1, l2] += 1

                smoothing_pred[prd_lbl] += 1.0
                count_pred += 1.0

        return fg_matrix, bg_matrix, (smoothing_pred/count_pred)

    def __getitem__(self, index):
        image_id = self.ids[index]
        targets = self.coco.dataset[image_id]

        LOG.debug(image_id)
        local_file_path = os.path.join(self.image_dir, image_id)
        with open(local_file_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_id,
            'local_file_path': local_file_path,
        }

        anns = []
        dict_counter = {}
        for target in targets:
            for type_obj in ['subject', 'object']:
                predicate = target['predicate']
                x = target[type_obj]['bbox'][2]
                y = target[type_obj]['bbox'][0]
                w = target[type_obj]['bbox'][3] - x
                h = target[type_obj]['bbox'][1] - y
                c = int(target[type_obj]['category'])

                x1 = target['object']['bbox'][2]
                y1 = target['object']['bbox'][0]
                w1 = target['object']['bbox'][3] - x1
                h1 = target['object']['bbox'][1] - y1
                c1 = int(target['object']['category'])
                if (x, y, w, h, c) in dict_counter:
                    if type_obj=='subject':
                        if (x1, y1, w1, h1, c1) in dict_counter:
                            anns[dict_counter[(x, y, w, h, c)]['detection_id']]['object_index'].append(dict_counter[(x1, y1, w1, h1, c1)]['detection_id'])
                        else:
                            anns[dict_counter[(x, y, w, h, c)]['detection_id']]['object_index'].append(len(anns))
                        anns[dict_counter[(x, y, w, h, c)]['detection_id']]['predicate'].append(predicate)
                else:
                    dict_counter[(x, y, w, h, c)] = {'detection_id': len(anns)}
                    object_index = [len(anns) + 1] if (type_obj=='subject') else []
                    if type_obj=='subject':
                        if (x1, y1, w1, h1, c1) in dict_counter:
                            object_index = [dict_counter[(x1, y1, w1, h1, c1)]['detection_id']]

                    anns.append({
                        'detection_id': len(anns),
                        'image_id': index,
                        'category_id': int(target[type_obj]['category']) + 1,
                        'bbox': [x, y, w, h],
                        "area": w*h,
                        "iscrowd": 0,
                        "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                        "segmentation":[],
                        'num_keypoints': 5,
                        'object_index': object_index,
                        'predicate': [predicate] if type_obj=='subject' else [],
                    })

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, meta)

        # transform image

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        LOG.debug(meta)

        return image, anns, meta

    def __len__(self):
        return len(self.ids)
