import numpy as np
from ..annotation import AnnotationRaf

class ToRafAnnotations:
    def __init__(self, obj_categories, rel_categories):
        self.obj_categories = obj_categories
        self.rel_categories = rel_categories

    def __call__(self, anns):
        annotations = []
        for ann in anns:
            if ann['iscrowd'] or not np.any(ann['bbox']) or not len(ann['object_index']) > 0:
                continue
            bbox = ann['bbox']
            category_id = ann['category_id']
            for object_id, predicate in zip(ann['object_index'], ann['predicate']):
                if anns[object_id]['iscrowd']:
                    continue
                bbox_object = anns[object_id]['bbox']
                object_category = anns[object_id]['category_id']
                annotations.append(AnnotationRaf(self.obj_categories,
                                    self.rel_categories).set(
                                    object_category, category_id,
                                    predicate+1, None, None, None, bbox, bbox_object
                                    ))

        return annotations
