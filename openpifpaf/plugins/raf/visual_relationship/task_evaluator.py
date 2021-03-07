"""
Coppied from Rowan Zellers, modified by Ji
"""
"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import numpy as np
import math

from tqdm import tqdm
from collections import defaultdict

from functools import reduce

np.set_printoptions(precision=3)

topk = 100

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def boxes_union(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    xmin = np.minimum(boxes1[:, 0], boxes2[:, 0])
    ymin = np.minimum(boxes1[:, 1], boxes2[:, 1])
    xmax = np.maximum(boxes1[:, 2], boxes2[:, 2])
    ymax = np.maximum(boxes1[:, 3], boxes2[:, 3])
    return np.vstack((xmin, ymin, xmax, ymax)).transpose()

def eval_rel_results(all_results):

    prd_k_set = [1]

    eval_sets = (False, )

    stats = defaultdict(lambda: defaultdict())
    for phrdet in eval_sets:
        eval_metric = 'phrdet' if phrdet else 'reldet'
        print('{}:'.format(eval_metric))

        for prd_k in prd_k_set:
            print('prd_k = {}:'.format(prd_k))

            recalls = {20: 0, 50: 0, 100: 0, -1:0}
            phrdet_acc = 0
            phrdet_acc_50 = 0

            all_gt_cnt = 0
            all_gt_match = 0
            all_gt_match_50 = 0
            all_gt_match_full = 0
            topk_dets = []
            for im_i, res in enumerate(tqdm(all_results)):

                # in oi_all_rel some images have no dets
                if len(res['prd_scores']) == 0:
                    det_boxes_s_top = np.zeros((0, 4), dtype=np.float32)
                    det_boxes_o_top = np.zeros((0, 4), dtype=np.float32)
                    det_labels_s_top = np.zeros(0, dtype=np.int32)
                    det_labels_p_top = np.zeros(0, dtype=np.int32)
                    det_labels_o_top = np.zeros(0, dtype=np.int32)
                    det_scores_top = np.zeros(0, dtype=np.float32)
                    det_labels_spo_top = np.vstack(
                        (det_labels_s_top, det_labels_p_top, det_labels_o_top)).transpose()
                    det_boxes_so_top = np.hstack(
                        (det_boxes_s_top, det_boxes_o_top))
                else:
                    det_boxes_sbj = res['sbj_boxes']  # (#num_rel, 4)
                    det_boxes_obj = res['obj_boxes']  # (#num_rel, 4)
                    det_labels_sbj = res['sbj_labels']  # (#num_rel,)
                    det_labels_obj = res['obj_labels']  # (#num_rel,)
                    det_scores_sbj = res['sbj_scores']  # (#num_rel,)
                    det_scores_obj = res['obj_scores']  # (#num_rel,)
                    det_scores_prd = res['prd_scores'].reshape(res['prd_scores'].shape[0],-1)

                    det_labels_prd = res['prd_labels'].reshape(res['prd_labels'].shape[0],-1)

                    det_scores_so = det_scores_sbj * det_scores_obj
                    det_scores_spo = det_scores_so[:, None] * det_scores_prd[:, :prd_k]
                    # det_scores_spo = det_scores_prd[:, :prd_k]
                    #det_scores_inds = argsort_desc(det_scores_spo)[:topk]
                    det_scores_inds = argsort_desc(det_scores_spo)
                    det_scores_top = det_scores_spo[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                    det_boxes_so_top = np.hstack(
                        (det_boxes_sbj[det_scores_inds[:, 0]], det_boxes_obj[det_scores_inds[:, 0]]))

                    det_labels_p_top = det_labels_prd[det_scores_inds[:, 0], det_scores_inds[:, 1]]

                    det_labels_spo_top = np.vstack(
                        (det_labels_sbj[det_scores_inds[:, 0]], det_labels_p_top, det_labels_obj[det_scores_inds[:, 0]])).transpose()

                    det_labels_spo_top_full = np.vstack(
                        (det_labels_sbj, det_labels_prd[:,0], det_labels_obj)).transpose()

#                     cand_inds = np.where(det_scores_top > cfg.TEST.SPO_SCORE_THRESH)[0]
#                     det_boxes_so_top = det_boxes_so_top[cand_inds]
#                     det_labels_spo_top = det_labels_spo_top[cand_inds]
#                     det_scores_top = det_scores_top[cand_inds]

                    det_boxes_s_top = det_boxes_so_top[:, :4]
                    det_boxes_o_top = det_boxes_so_top[:, 4:]
                    det_labels_s_top = det_labels_spo_top[:, 0]
                    det_labels_p_top = det_labels_spo_top[:, 1]
                    det_labels_o_top = det_labels_spo_top[:, 2]

                topk_dets.append(dict(image=res['image'],
                                      det_boxes_s_top=det_boxes_s_top,
                                      det_boxes_o_top=det_boxes_o_top,
                                      det_labels_s_top=det_labels_s_top,
                                      det_labels_p_top=det_labels_p_top,
                                      det_labels_o_top=det_labels_o_top,
                                      det_scores_top=det_scores_top))

                gt_boxes_sbj = res['gt_sbj_boxes']  # (#num_gt, 4)
                gt_boxes_obj = res['gt_obj_boxes']  # (#num_gt, 4)
                gt_labels_sbj = res['gt_sbj_labels']  # (#num_gt,)
                gt_labels_obj = res['gt_obj_labels']  # (#num_gt,)
                gt_labels_prd = res['gt_prd_labels']  # (#num_gt,)
                gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
                gt_labels_spo = np.vstack((gt_labels_sbj, gt_labels_prd, gt_labels_obj)).transpose()
                # Compute recall. It's most efficient to match once and then do recall after
                # det_boxes_so_top is (#num_rel, 8)
                # det_labels_spo_top is (#num_rel, 3)
                if phrdet:
                    det_boxes_r_top = boxes_union(det_boxes_s_top, det_boxes_o_top)
                    gt_boxes_r = boxes_union(gt_boxes_sbj, gt_boxes_obj)
                    pred_to_gt = _compute_pred_matches(
                        gt_labels_spo, det_labels_spo_top,
                        gt_boxes_r, det_boxes_r_top,
                        phrdet=phrdet)
                else:
                    pred_to_gt = _compute_pred_matches(
                        gt_labels_spo, det_labels_spo_top,
                        gt_boxes_so, det_boxes_so_top,
                        phrdet=phrdet)
                gt_has_match = intersect_2d(gt_labels_spo, det_labels_spo_top[:100]).any(1)
                gt_has_match_50 = intersect_2d(gt_labels_spo, det_labels_spo_top[:50]).any(1)
                gt_has_match_full = intersect_2d(gt_labels_spo, det_labels_spo_top_full).any(1)
                all_gt_match += sum(gt_has_match)
                all_gt_match_50 += sum(gt_has_match_50)
                all_gt_match_full += sum(gt_has_match_full)
                all_gt_cnt += gt_labels_spo.shape[0]
                for k in recalls:
                    if len(pred_to_gt):
                        if k == -1:
                            match = reduce(np.union1d, pred_to_gt)
                        else:
                            match = reduce(np.union1d, pred_to_gt[:k])
                    else:
                        match = []
                    recalls[k] += len(match)

                topk_dets[-1].update(dict(gt_boxes_sbj=gt_boxes_sbj,
                                          gt_boxes_obj=gt_boxes_obj,
                                          gt_labels_sbj=gt_labels_sbj,
                                          gt_labels_obj=gt_labels_obj,
                                          gt_labels_prd=gt_labels_prd))


            for k in recalls:
                recalls[k] = float(recalls[k]) / (float(all_gt_cnt) + 1e-12)
            phrdet_acc = all_gt_match/all_gt_cnt
            phrdet_acc_50 = all_gt_match_50/all_gt_cnt
            phrdet_acc_full = all_gt_match_full/all_gt_cnt
                #print_stats(recalls)

        stats[eval_metric]['recall'] = dict(recalls)
        stats[eval_metric]['phrdet_acc_50'] = phrdet_acc_50
        stats[eval_metric]['phrdet_acc_100'] = phrdet_acc
        stats[eval_metric]['phrdet_acc_full'] = phrdet_acc_full
    return stats

def print_stats(recalls):
    # print('====================== ' + 'sgdet' + ' ============================')
    for k, v in recalls.items():
        print('R@%i: %f' % (k, v))

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def log_average_miss_rate(prec, rec, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

def eval_det_results(det_pred, det_gt, gt_classes, gt_counter_per_class):
    tp_perclass = {}
    fp_perclass = {}
    recall_perclass = {}
    n_classes = len(gt_classes)
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    count_true_positives = {}
    for class_i, class_name in gt_classes.items():
        count_true_positives[class_name] = 0
        dr_data = det_pred[class_i]
        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            image_id = detection[0]

            ground_truth_data = det_gt[image_id]
            ovmax = -1
            gt_match = -1
            bb = [ float(detection[1][0]), float(detection[1][1]), float(detection[1][0]+detection[1][2]), float(detection[1][1]+detection[1][3])]

            for obj in ground_truth_data:
                # look for a class_name match
                if obj[0] == class_i:
                    bbgt = [float(obj[1][0]), float(obj[1][1]), float(obj[1][0]+obj[1][2]), float(obj[1][1]+obj[1][3])]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            min_overlap = 0.5
            if ovmax >= min_overlap:
                if not bool(gt_match[-1]):
                    # true positive
                    tp[idx] = 1
                    gt_match[-1] = True
                    count_true_positives[class_name] += 1
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1

        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        #print(tp)

        tp_perclass[class_name] =  0
        fp_perclass[class_name] = 0
        if len(tp)>0:
            tp_perclass[class_name] = tp[-1]
        if len(fp)>0:
            fp_perclass[class_name] = fp[-1]
        recall_perclass[class_name] = float(tp_perclass[class_name])/gt_counter_per_class[class_name]
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap

        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]

        ap_dictionary[class_name] = ap

        # n_images = counter_images_per_class[class_name]
        # lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
        # lamr_dictionary[class_name] = lamr
    mAP = sum_AP / n_classes
    return mAP, tp_perclass, fp_perclass, recall_perclass


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh=0.5, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            gt_box = gt_box.astype(dtype=np.float32, copy=False)
            boxes = boxes.astype(dtype=np.float32, copy=False)
            rel_iou = bbox_overlaps(gt_box[None, :], boxes)[0]

            inds = rel_iou >= iou_thresh
        else:
            gt_box = gt_box.astype(dtype=np.float32, copy=False)
            boxes = boxes.astype(dtype=np.float32, copy=False)
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))
