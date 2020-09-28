import os
import copy
import logging
import numpy as np
import torch.utils.data
import torchvision
import json

from PIL import Image
from openpifpaf import transforms, utils

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))

class VisualRelationship(torch.utils.data.Dataset):


    def __init__(self, image_dir, ann_file, *,
                 n_images=None, preprocess=None,
                 category_ids=None):

        self.root = image_dir
        self.imgs = [(os.path.join(self.root, k),v) for k,v in json.load(open(ann_file)).items()]


        if n_images:
            self.imgs = self.imgs[:n_images]

        print('Images: {}'.format(len(self.imgs)))

        # PifPaf
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        chosen_img = self.imgs[index]
        img_path = chosen_img[0]
        with open(os.path.join(img_path), 'rb') as f:
            image = Image.open(f).convert('RGB')

        initial_size = image.size
        meta_init = {
            'dataset_index': index,
            'image_id': index,
            'file_dir': img_path,
            'file_name': os.path.basename(img_path),
        }

        anns = []
        dict_counter = {}
        for target in chosen_img[1]:
            for type_obj in ['subject', 'object']:
                predicate = target['predicate']
                x = target[type_obj]['bbox'][2]
                y = target[type_obj]['bbox'][0]
                w = target[type_obj]['bbox'][3] - x
                h = target[type_obj]['bbox'][1] - y

                x1 = target['object']['bbox'][2]
                y1 = target['object']['bbox'][0]
                w1 = target['object']['bbox'][3] - x1
                h1 = target['object']['bbox'][1] - y1
                if (x, y, w, h) in dict_counter:
                    if type_obj=='subject':
                        if (x1, y1, w1, h1) in dict_counter:
                            anns[dict_counter[(x, y, w, h)]['detection_id']]['object_index'].append(dict_counter[(x1, y1, w1, h1)]['detection_id'])
                        else:
                            anns[dict_counter[(x, y, w, h)]['detection_id']]['object_index'].append(len(anns))
                        anns[dict_counter[(x, y, w, h)]['detection_id']]['predicate'].append(predicate)
                else:
                    object_index = [len(anns) + 1] if (type_obj=='subject') else []
                    if type_obj=='subject':
                        if (x1, y1, w1, h1) in dict_counter:
                            object_index = [dict_counter[(x1, y1, w1, h1)]['detection_id']]
                    dict_counter[(x, y, w, h)] = {'detection_id': len(anns)}
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
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)

        # transform image

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        LOG.debug(meta)

        return image, anns, meta

    def __len__(self):
        return len(self.imgs)

    def write_evaluations(self, eval_class, path, total_time):
        pass
