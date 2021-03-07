from openpifpaf.encoder import AnnRescalerDet
import numpy as np
import copy

import math

def mask_valid_area_offset(intensities, valid_area, *, fill_value=0):
    """Mask area.

    Intensities is either a feature map or an image.
    """
    if valid_area is None:
        return

    if valid_area[1] >= 1.0:
        intensities[:int(valid_area[1]), :] = fill_value
    if valid_area[0] >= 1.0:
        intensities[:, :int(valid_area[0])] = fill_value

    max_i = int(math.ceil(valid_area[1] + valid_area[3])) + 1
    max_j = int(math.ceil(valid_area[0] + valid_area[2])) + 1
    if 0 < max_i < intensities.shape[0]:
        intensities[max_i:, :] = fill_value
    if 0 < max_j < intensities.shape[1]:
        intensities[:, max_j:] = fill_value

class AnnRescalerRel(AnnRescalerDet):
    def bg_mask(self, anns, width_height, *, crowd_margin):
        """Create background mask taking crowd annotations into account."""
        # mask = np.ones((
        #     self.n_categories,
        #     (width_height[1]) // self.stride,
        #     (width_height[0]) // self.stride,
        # ), dtype=np.bool)
        #
        # mask_offset = np.ones((
        #     self.n_categories,
        #     (width_height[1]) // (self.stride*2),
        #     (width_height[0]) // (self.stride*2),
        # ), dtype=np.bool)

        mask = np.ones((
            self.n_categories,
            (width_height[1] - 1) // self.stride + 1,
            (width_height[0] - 1) // self.stride + 1,
        ), dtype=np.bool)

        mask_offset = np.ones((
            self.n_categories,
            (width_height[1]-1) // (self.stride*2)+1,
            (width_height[0]-1) // (self.stride*2)+1,
        ), dtype=np.bool)

        return mask, mask_offset

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
    def valid_area_offset(self, meta):
        if 'valid_area' not in meta:
            return None

        return (
            meta['valid_area'][0] / (self.stride*2),
            meta['valid_area'][1] / (self.stride*2),
            meta['valid_area'][2] / (self.stride*2),
            meta['valid_area'][3] / (self.stride*2),
        )
