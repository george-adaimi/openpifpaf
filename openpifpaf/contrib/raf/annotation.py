import numpy as np
from openpifpaf.annotation import Base

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

    def set(self, category_id_obj, category_id_sub, category_id_rel, score_sub, score_rel, score_obj, bbox_sub, bbox_obj):
        """Set score to None for a ground truth annotation."""
        self.category_id_obj = category_id_obj
        self.category_id_sub = category_id_sub
        self.category_id_rel = category_id_rel
        self.score_sub = score_sub
        self.score_obj = score_obj
        self.score_rel = score_rel
        self.bbox_obj = np.asarray(bbox_obj)
        self.bbox_sub = np.asarray(bbox_sub)
        return self

    @property
    def category_sub(self):
        return self.obj_categories[self.category_id_sub - 1]
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
            'score_sub': max(0.001, round(float(self.score_sub), 3)),
            'score_obj': max(0.001, round(float(self.score_obj), 3)),
            'score_rel': max(0.001, round(float(self.score_rel), 3)),
            'bbox_sub': [round(float(c), 2) for c in self.bbox_sub],
            'bbox_obj': [round(float(c), 2) for c in self.bbox_obj],
            'bbox_rel': [round(float(c), 2) for c in self.bbox_rel],
        }
