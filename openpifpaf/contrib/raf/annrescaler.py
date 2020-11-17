from openpifpaf.encoder import AnnRescalerDet
import numpy as np
import copy

class AnnRescalerRel(AnnRescalerDet):
    def bg_mask(self, anns, width_height, *, crowd_margin):
        """Create background mask taking crowd annotations into account."""
        mask = np.ones((
            self.n_categories,
            (width_height[1] - 1) // self.stride + 1,
            (width_height[0] - 1) // self.stride + 1,
        ), dtype=np.bool)

        return mask

    def relations(self, anns):
        dict_temp = {}
        for ann in anns:
            dict_temp[ann['detection_id']] = copy.deepcopy(ann)
            dict_temp[ann['detection_id']]['bbox'] = dict_temp[ann['detection_id']]['bbox'] / self.stride

        for k, ann in dict_temp.items():
            ann['predicate'] = [other_pred for other_ind, other_pred in zip(ann['object_index'], ann['predicate']) if other_ind in dict_temp.keys()]
            ann['object_index'] = [other_ind for other_ind in ann['object_index'] if other_ind in dict_temp.keys()]

        #category_bboxes = [(ann['category_id'], ann['bbox'] / self.stride, dict_temp[ann['object_index']], ann['predicate'])
        #                   for k, ann in dict_temp.items() if (ann['object_index'] in dict_temp and (not (ann['iscrowd'] or dict_temp[ann['object_index']]['iscrowd'])))]
        return dict_temp
