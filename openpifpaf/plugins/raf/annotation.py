import numpy as np
from openpifpaf.annotation import Base
import copy


class AnnotationRaf(Base):
    def __init__(self, obj_categories, rel_categories):
        self.obj_categories = obj_categories
        self.rel_categories = rel_categories
        self.category_id_obj = None
        self.category_id_sub = None
        self.category_id_rel = None
        self.score_sub = None
        self.score_obj = None
        self.score_rel = None
        self.bbox_obj = None
        self.bbox_sub = None
        self.idx_subj = None
        self.idx_obj = None

    def set(self, category_id_obj, category_id_sub, category_id_rel, score_sub,
            score_rel, score_obj, bbox_sub, bbox_obj, idx_subj, idx_obj):
        """Set score to None for a ground truth annotation."""
        self.category_id_obj = category_id_obj
        self.category_id_sub = category_id_sub
        self.category_id_rel = category_id_rel
        self.score_sub = score_sub
        self.score_obj = score_obj
        self.score_rel = score_rel
        self.bbox_obj = np.asarray(bbox_obj)
        self.bbox_sub = np.asarray(bbox_sub)
        self.idx_subj = idx_subj
        self.idx_obj = idx_obj
        return self

    @property
    def category_sub(self):
        return self.obj_categories[self.category_id_sub - 1]

    @property
    def score(self):
        return self.score_sub*self.score_obj*self.score_rel

    @property
    def category_obj(self):
        return self.obj_categories[self.category_id_obj - 1]

    @property
    def category_rel(self):
        return self.rel_categories[self.category_id_rel - 1]

    def json_data(self):
        return {
            'category_id_sub': self.category_id_sub,
            'category_id_obj': self.category_id_obj,
            'category_id_rel': self.category_id_rel,
            'category_sub': self.category_sub,
            'category_obj': self.category_obj,
            'category_rel': self.category_rel,
            'score_sub': max(0.001, round(float(self.score_sub), 3)) if not self.score_sub is None else self.score_sub,
            'score_obj': max(0.001, round(float(self.score_obj), 3)) if not self.score_obj is None else self.score_obj,
            'score_rel': max(0.001, round(float(self.score_rel), 3)) if not self.score_rel is None else self.score_rel,
            'bbox_sub': [round(float(c), 2) for c in self.bbox_sub],
            'bbox_obj': [round(float(c), 2) for c in self.bbox_obj],
        }

    def inverse_transform(self, meta):

        ann = copy.deepcopy(self)

        angle = -meta['rotation']['angle']
        if angle != 0.0:
            rw = meta['rotation']['width']
            rh = meta['rotation']['height']
            ann.bbox_sub = utils.rotate_box(ann.bbox_sub, rw - 1, rh - 1, angle)
            ann.bbox_obj = utils.rotate_box(ann.bbox_obj, rw - 1, rh - 1, angle)

        ann.bbox_sub[:2] += meta['offset']
        ann.bbox_sub[:2] /= meta['scale']
        ann.bbox_sub[2:] /= meta['scale']

        ann.bbox_obj[:2] += meta['offset']
        ann.bbox_obj[:2] /= meta['scale']
        ann.bbox_obj[2:] /= meta['scale']

        if meta['hflip']:
            w = meta['width_height'][0]
            ann.bbox_sub[0] = -(ann.bbox_sub[0] + ann.bbox_sub[2]) - 1.0 + w
            ann.bbox_obj[0] = -(ann.bbox_obj[0] + ann.bbox_obj[2]) - 1.0 + w

        return ann
