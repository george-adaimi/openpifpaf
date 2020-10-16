import numpy as np
from openpifpaf.annotation import Base

class AnnotationRaf(Base):
    def __init__(self, obj_categories, rel_categories):
        self.obj_categories = obj_categories
        self.rel_categories = rel_categories
        self.category_id_obj = None
        self.category_id_sub = None
        self.category_id_rel = None
        self.score = None
        self.bbox_obj = None
        self.bbox_subj = None

    def set(self, category_id_obj, category_id_sub, score, bbox, ):
        """Set score to None for a ground truth annotation."""
        self.category_id = category_id
        self.score = score
        self.bbox = np.asarray(bbox)
        return self

    @property
    def category(self):
        return self.categories[self.category_id - 1]

    def json_data(self):
        return {
            'category_id': self.category_id,
            'category': self.category,
            'score': max(0.001, round(float(self.score), 3)),
            'bbox': [round(float(c), 2) for c in self.bbox],
        }
