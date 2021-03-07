import task_evaluator
from scipy.io import loadmat
#import tables
import numpy as np
import copy

file1 = loadmat('relationship_det_result.mat')
#file1 = loadmat('predicate_det_result.mat')
rlp_confs = file1['rlp_confs_ours'][0]
rlp_labels = file1['rlp_labels_ours'][0]
rlp_sub_bboxes = file1['sub_bboxes_ours'][0]
rlp_obj_bboxes = file1['obj_bboxes_ours'][0]

file2 = loadmat('gt.mat')
gt_labels = file2['gt_tuple_label'][0]
gt_sub_bboxes = file2['gt_sub_bboxes'][0]
gt_obj_bboxes = file2['gt_obj_bboxes'][0]

roidb_pred = []
for img_idx, (rlp_conf, rlp_label, rlp_sub_bbox, rlp_obj_bboxes, gt_label, gt_sub_bbox, gt_obj_bbox) in enumerate(zip(rlp_confs, rlp_labels, rlp_sub_bboxes, rlp_obj_bboxes, gt_labels, gt_sub_bboxes, gt_obj_bboxes)):
    image_annotations = {}
    image_annotations['image'] = img_idx
    if len(rlp_label) != 0:
        image_annotations['sbj_boxes'] = np.asarray(rlp_sub_bbox)
        image_annotations['obj_boxes'] = np.asarray(rlp_obj_bboxes)
        image_annotations['sbj_labels'] = np.asarray(rlp_label[:,0]-1)
        image_annotations['obj_labels'] = np.asarray(rlp_label[:,2]-1)
        image_annotations['prd_labels'] = np.asarray(rlp_label[:,1]-1)

        image_annotations['sbj_scores'] = np.cbrt(np.asarray(rlp_conf[:,0]))
        image_annotations['obj_scores'] = np.cbrt(np.asarray(rlp_conf[:,0]))
        image_annotations['prd_scores'] = np.cbrt(np.asarray(rlp_conf[:,0]))
    else:
        image_annotations['sbj_boxes'] = np.zeros((0, 4), dtype=np.float32)
        image_annotations['obj_boxes'] = np.zeros((0, 4), dtype=np.float32)
        image_annotations['sbj_labels'] = np.zeros(0, dtype=np.int32)
        image_annotations['obj_labels'] = np.zeros(0, dtype=np.int32)
        image_annotations['prd_labels'] = np.zeros(0, dtype=np.int32)

        image_annotations['sbj_scores'] = np.zeros(0, dtype=np.int32)
        image_annotations['obj_scores'] = np.zeros(0, dtype=np.int32)
        image_annotations['prd_scores'] = np.zeros(0, dtype=np.int32)

    if len(gt_label) != 0:
        image_annotations['gt_sbj_boxes'] = np.asarray(gt_sub_bbox)
        image_annotations['gt_obj_boxes'] = np.asarray(gt_obj_bbox)
        image_annotations['gt_sbj_labels'] = np.asarray(gt_label[:,0]-1)
        image_annotations['gt_obj_labels'] = np.asarray(gt_label[:,2]-1)
        image_annotations['gt_prd_labels'] = np.asarray(gt_label[:,1]-1)
    else:
        image_annotations['gt_sbj_boxes'] = np.zeros((0, 4), dtype=np.float32)
        image_annotations['gt_obj_boxes'] = np.zeros((0, 4), dtype=np.float32)
        image_annotations['gt_sbj_labels'] = np.zeros(0, dtype=np.int32)
        image_annotations['gt_obj_labels'] = np.zeros(0, dtype=np.int32)
        image_annotations['gt_prd_labels'] = np.zeros(0, dtype=np.int32)

    roidb_pred.append(copy.deepcopy(image_annotations))

recalls = task_evaluator.eval_rel_results(roidb_pred)
print(recalls)
