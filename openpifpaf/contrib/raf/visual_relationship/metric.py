import logging
import os
import numpy as np
import zipfile
from tqdm import tqdm
from PIL import Image

from openpifpaf.metric.base import Base
from . import task_evaluator

LOG = logging.getLogger(__name__)

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class VRD(Base):
    text_labels = ['R@20', 'R@50', 'R@100']

    def __init__(self):

        self.roidb_pred = []
        self.image_ids = []


    def accumulate(self, predictions, image_meta, ground_truth=None):
        image_id = int(image_meta['image_id'])
        width, height = image_meta['width_height']
        self.image_ids.append(image_id)

        obj = []
        rela = []
        sub = []
        sub_bbox = []
        obj_bbox = []

        image_annotations = {}
        image_annotations['image'] = image_meta['file_dir']
        image_annotations['width'] = width
        image_annotations['height'] = height
        for gt_ann in ground_truth:
            pred_data = gt_ann.json_data()
            sub.append(pred_data['category_id_sub']-1)
            obj.append(pred_data['category_id_obj']-1)
            rela.append(pred_data['category_id_rel']-1)
            x,y,w,h = pred_data['bbox_sub']
            x1, x2 = np.clip([x, x+w], a_min=0, a_max=width)
            y1, y2 = np.clip([y, y+h], a_min=0, a_max=height)
            sub_bbox.append([x1, y1, x2, y2])
            x,y,w,h = pred_data['bbox_obj']
            x1, x2 = np.clip([x, x+w], a_min=0, a_max=width)
            y1, y2 = np.clip([y, y+h], a_min=0, a_max=height)
            obj_bbox.append([x1, y1, x2, y2])

        if len(ground_truth) == 0:
            obj = np.zeros(0, dtype=np.int32)
            rela = np.zeros(0, dtype=np.int32)
            sub = np.zeros(0, dtype=np.int32)
            sub_bbox = np.zeros((0, 4), dtype=np.float32)
            obj_bbox = np.zeros((0, 4), dtype=np.float32)
        image_annotations['gt_sbj_boxes'] = np.asarray(sub_bbox)
        image_annotations['gt_obj_boxes'] = np.asarray(obj_bbox)
        image_annotations['gt_sbj_labels'] = np.asarray(sub)
        image_annotations['gt_obj_labels'] = np.asarray(obj)
        image_annotations['gt_prd_labels'] = np.asarray(rela)

        obj = []
        rela = []
        sub = []
        obj_scores = [] #np.ones(len(ground_truth), dtype=np.int32)
        rela_scores = [] #np.ones(len(ground_truth), dtype=np.int32)
        sub_scores = [] #np.ones(len(ground_truth), dtype=np.int32)
        sub_bbox = []
        obj_bbox = []

        for pred in predictions:
            pred_data = pred.json_data()
            sub.append(pred_data['category_id_sub']-1)
            obj.append(pred_data['category_id_obj']-1)
            rela.append(pred_data['category_id_rel']-1)
            x,y,w,h = pred_data['bbox_sub']
            x1, x2 = np.clip([x, x+w], a_min=0, a_max=width)
            y1, y2 = np.clip([y, y+h], a_min=0, a_max=height)
            sub_bbox.append([x1, y1, x2, y2])
            x,y,w,h = pred_data['bbox_obj']
            x1, x2 = np.clip([x, x+w], a_min=0, a_max=width)
            y1, y2 = np.clip([y, y+h], a_min=0, a_max=height)
            obj_bbox.append([x1, y1, x2, y2])
            sub_scores.append(pred_data['score_sub'])
            rela_scores.append(pred_data['score_rel'])
            obj_scores.append(pred_data['score_obj'])

        image_annotations['sbj_boxes'] = np.asarray(sub_bbox)
        image_annotations['obj_boxes'] = np.asarray(obj_bbox)
        image_annotations['sbj_labels'] = np.asarray(sub)
        image_annotations['obj_labels'] = np.asarray(obj)
        image_annotations['prd_labels'] = np.asarray(rela)
        image_annotations['sbj_scores'] = np.asarray(sub_scores)
        image_annotations['obj_scores'] = np.asarray(obj_scores)
        image_annotations['prd_scores'] = np.asarray(rela_scores)
        self.roidb_pred.append(image_annotations)


    def write_predictions(self, filename, additional_data=None):
        mkdir_if_missing(filename)
        np.save(os.path.join(filename, 'rela_prediction.npy'),self.roidb_pred)

        LOG.info('wrote predictions to %s', filename)

        if additional_data:
            with open(os.path.join(filename, 'predictions.pred_meta.json'), 'w') as f:
                json.dump(additional_data, f)
            LOG.info('wrote %s.pred_meta.json', filename)


    def stats(self):
        recalls = task_evaluator.eval_rel_results(self.roidb_pred)

        data = {
            'stats': [recalls[20], recalls[50], recalls[100]],
            'text_labels': self.text_labels,
        }

        return data
