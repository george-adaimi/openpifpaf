from openpifpaf.metric.base import Base

class VG(Base):
    def __init__(self, gt_dir, img_dir, mode, iou_types=['bbox', 'relations']):
        assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}
        self.iou_types = iou_types
        self.bbox_anns_gt = []
        self.bbox_anns_pred = []
        self.rel_anns = []

        if 'relations' in iou_types:
            result_dict = {}
            self.evaluator = {}
            # tradictional Recall@K
            eval_recall = SGRecall(result_dict)
            eval_recall.register_container(mode)
            self.evaluator['eval_recall'] = eval_recall

            # no graphical constraint
            eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
            eval_nog_recall.register_container(mode)
            self.evaluator['eval_nog_recall'] = eval_nog_recall

            # test on different distribution
            eval_zeroshot_recall = SGZeroShotRecall(result_dict)
            eval_zeroshot_recall.register_container(mode)
            self.evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

            # test on no graph constraint zero-shot recall
            eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict)
            eval_ng_zeroshot_recall.register_container(mode)
            self.evaluator['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall

            # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
            eval_pair_accuracy = SGPairAccuracy(result_dict)
            eval_pair_accuracy.register_container(mode)
            self.evaluator['eval_pair_accuracy'] = eval_pair_accuracy

            # used for meanRecall@K
            eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
            eval_mean_recall.register_container(mode)
            self.evaluator['eval_mean_recall'] = eval_mean_recall

            # used for no graph constraint mean Recall@K
            eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
            eval_ng_mean_recall.register_container(mode)
            self.evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

            # prepare all inputs
            self.global_container = {}
            self.global_container['zeroshot_triplet'] = zeroshot_triplet
            self.global_container['result_dict'] = result_dict
            self.global_container['mode'] = mode
            self.global_container['multiple_preds'] = multiple_preds
            self.global_container['num_rel_category'] = num_rel_category
            self.global_container['iou_thres'] = iou_thres
            self.global_container['attribute_on'] = attribute_on
            self.global_container['num_attributes'] = num_attributes

    def accumulate(self, predictions, image_meta, ground_truth=None):
        predicates_rel, predictions_det = predictions
        image_id = int(image_meta['image_id'])
        width, height = image_meta['width_height']
        self.image_ids.append(image_id)

        if 'bbox' in iou_types:
            for pred in ground_truth:
                self.bbox_anns.append({
                    'area': pred['area'],
                    'bbox': pred['bbox'], # xywh
                    'category_id': pred['category_id'],
                    'id': len(self.bbox_anns),
                    'image_id': image_id,
                    'iscrowd': 0,
                })

            box = []
            score = []
            label = []
            for pred in predictions_det:
                box.append(pred.bbox)
                score.append(pred.score)
                label.append(pred.category_id)

            image_id_pred = np.asarray([image_id]*len(box))
            self.bbox_anns_pred.append(
                np.column_stack((image_id_pred, box, score, label))
            )

        if 'relations' in iou_types:
            for pred in predictions_rel:
                self.rel_anns.append({
                    'area': pred.bbox[2]*pred.bbox[3],
                    'bbox': pred.bbox, # xywh
                    'category_id': pred.category_id,
                    'id': len(self.bbox_anns),
                    'image_id': image_id,
                    'iscrowd': 0,
                })

            self.evaluate_relation_of_one_image(ground_truth, predicates_rel, self.global_container, self.evaluator)
    def stats_det(self):
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in range(len(self.image_ids))],
            'categories': [
                {'supercategory': 'person', 'id': i, 'name': name}
                for i, name in enumerate(dataset.ind_to_classes) if name != '__background__'
                ],
            'annotations': self.bbox_anns_gt,
        }
        fauxcoco.createIndex()

        cocolike_predictions = np.concatenate(self.bbox_anns_pred, 0)
        # evaluate via coco API
        res = fauxcoco.loadRes(cocolike_predictions)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        coco_eval.params.imgIds = list(range(len(self.image_ids)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    def stats_rel(self):

    def stats(self):

    def evaluate_relation_of_one_image(self, groundtruth, prediction, global_container, evaluator):
        """
        Returns:
            pred_to_gt: Matching from predicate to GT
            pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
            pred_triplet_scores: [cls_0score, relscore, cls1_score]
        """
        #unpack all inputs
        mode = global_container['mode']

        local_container = {}
        local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

        # if there is no gt relations for current image, then skip it
        if len(local_container['gt_rels']) == 0:
            return

        local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()                   # (#gt_objs, 4)
        local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()           # (#gt_objs, )

        # about relations
        local_container['pred_rel_inds'] = prediction.get_field('rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
        local_container['rel_scores'] = prediction.get_field('pred_rel_scores').detach().cpu().numpy()          # (#pred_rels, num_pred_class)

        # about objects
        local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()                  # (#pred_objs, 4)
        local_container['pred_classes'] = prediction.get_field('pred_labels').long().detach().cpu().numpy()     # (#pred_objs, )
        local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()              # (#pred_objs, )


        # to calculate accuracy, only consider those gt pairs
        # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
        # for sgcls and predcls
        if mode != 'sgdet':
            evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

        # to calculate the prior label based on statistics
        evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
        evaluator['eval_ng_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

        if mode == 'predcls':
            local_container['pred_boxes'] = local_container['gt_boxes']
            local_container['pred_classes'] = local_container['gt_classes']
            local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

        elif mode == 'sgcls':
            if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
                print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
        elif mode == 'sgdet' or mode == 'phrdet':
            pass
        else:
            raise ValueError('invalid mode')
        """
        elif mode == 'preddet':
            # Only extract the indices that appear in GT
            prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
            if prc.size == 0:
                for k in result_dict[mode + '_recall']:
                    result_dict[mode + '_recall'][k].append(0.0)
                return None, None, None
            pred_inds_per_gt = prc.argmax(0)
            pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
            rel_scores = rel_scores[pred_inds_per_gt]
            # Now sort the matching ones
            rel_scores_sorted = argsort_desc(rel_scores[:,1:])
            rel_scores_sorted[:,1] += 1
            rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))
            matches = intersect_2d(rel_scores_sorted, gt_rels)
            for k in result_dict[mode + '_recall']:
                rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
                result_dict[mode + '_recall'][k].append(rec_i)
            return None, None, None
        """

        if local_container['pred_rel_inds'].shape[0] == 0:
            return

        # Traditional Metric with Graph Constraint
        # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
        local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

        # No Graph Constraint
        evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
        # GT Pair Accuracy
        evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
        # Mean Recall
        evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
        # No Graph Constraint Mean Recall
        evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
        # Zero shot Recall
        evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
        # No Graph Constraint Zero-Shot Recall
        evaluator['eval_ng_zeroshot_recall'].calculate_recall(global_container, local_container, mode)

        return
