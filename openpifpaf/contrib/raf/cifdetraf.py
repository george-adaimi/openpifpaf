import argparse
from collections import defaultdict
import heapq
import logging
import time
import math
import copy
from typing import List

import heapq
import numpy as np

from openpifpaf.annotation import AnnotationDet
from .annotation import AnnotationRaf
from .raf_analyzer import RafAnalyzer
from openpifpaf.decoder import Decoder, utils
from .headmeta import Raf
from openpifpaf import headmeta,visualizer
from . import visualizer as visualizer_raf

from openpifpaf.functional import grow_connection_blend
LOG = logging.getLogger(__name__)

class CifDetRaf(Decoder):
    """Generate CifDetRaf from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    connection_method = 'blend'
    occupancy_visualizer = visualizer.Occupancy()
    force_complete = False
    greedy = False
    keypoint_threshold = 0.0
    nms = True

    def __init__(self,
                cifdet_metas: List[headmeta.CifDet],
                raf_metas: List[Raf],
                *,
                cifdet_visualizers=None,
                raf_visualizers=None):
        super().__init__()
        self.cifdet_metas = cifdet_metas
        self.raf_metas = raf_metas

        self.cifdet_visualizers = cifdet_visualizers
        if self.cifdet_visualizers is None:
            self.cifdet_visualizers = [visualizer.CifDet(meta) for meta in cifdet_metas]
        self.raf_visualizers = raf_visualizers
        if self.raf_visualizers is None:
            self.raf_visualizers = [visualizer_raf.Raf(meta) for meta in raf_metas]

        if self.nms is True:
            self.nms = utils.nms.Detection()


        self.confidence_scales = raf_metas[0].decoder_confidence_scales

        self.timers = defaultdict(float)

        # init by_target and by_source
        self.by_target = defaultdict(dict)

        self.by_source = defaultdict(dict)
        for j1 in range(len(self.raf_metas[0].obj_categories)):
            for j2 in range(len(self.raf_metas[0].obj_categories)):
                for raf_i in range(len(self.raf_metas[0].rel_categories)):
                    self.by_source[j1][j2] = (raf_i, True)
                    self.by_source[j2][j1] = (raf_i, True)

    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        return [
            CifDetRaf([meta], [meta_next])
            for meta, meta_next in zip(head_metas[:-1], head_metas[1:])
            if (isinstance(meta, headmeta.CifDet)
                and isinstance(meta_next, Raf))
        ]
    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        # check consistency
        assert args.seed_threshold >= args.keypoint_threshold

        cls.keypoint_threshold = (args.keypoint_threshold)
        cls.greedy = args.greedy
        cls.connection_method = args.connection_method

    def __call__(self, fields, initial_annotations=None, meta=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        for vis, meta in zip(self.cifdet_visualizers, self.cifdet_metas):
            vis.predicted(fields[meta.head_index])
        for vis, meta in zip(self.raf_visualizers, self.raf_metas):
            vis.predicted(fields[meta.head_index])

        cifdethr = utils.CifDetHr().fill(fields, self.cifdet_metas)
        seeds = utils.CifDetSeeds(cifdethr.accumulated).fill(fields, self.cifdet_metas)

        raf_analyzer = RafAnalyzer(cifdethr.accumulated).fill(fields, self.raf_metas)

        occupied = utils.Occupancy(cifdethr.accumulated.shape, 2, min_scale=4)
        annotations_det = []


        def mark_occupied(ann):
            for joint_i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue

                width = ann.joint_scales[joint_i]
                occupied.set(joint_i, xyv[0], xyv[1], width)  # width = 2 * sigma

        for v, f, x, y, w, h in seeds.get():
            if occupied.get(f, x, y):
                continue
            ann = AnnotationDet(self.cifdet_metas[0].categories).set(f + 1, v, (x - w/2.0, y - h/2.0, w, h))
            annotations_det.append(ann)
            #mark_occupied(ann)
            occupied.set(f, x, y, 0.1 * min(w, h))
        dict_rel = {}

        if self.nms is not None:
            annotations_det = self.nms.annotations(annotations_det)

        annotations = []
        for raf_v, index_s, x_s, y_s, raf_i, index_o, x_o, y_o in raf_analyzer.triplets:
            s_idx = None
            o_idx = None
            min_value_s = None
            min_value_o = None
            for ann_idx, ann in enumerate(annotations_det):
                if not(ann.category_id-1 == index_s or ann.category_id-1 == index_o):
                    continue
                a = ann.bbox[0] + ann.bbox[2]/2.0
                b = ann.bbox[1] + ann.bbox[3]/2.0
                curr_dist = (1/(raf_v*ann.score+0.00001))*(math.sqrt((a - x_s)**2+(b - y_s)**2))
                if min_value_s is None or curr_dist<min_value_s:
                    min_value_s = curr_dist
                    s_idx = ann_idx
                curr_dist = (1/(raf_v*ann.score+0.00001))*(math.sqrt((a - x_o)**2+(b - y_o)**2))
                if min_value_o is None or curr_dist<min_value_o:
                    min_value_o = curr_dist
                    o_idx = ann_idx
            if (s_idx, raf_i, o_idx) in dict_rel:
                annotations[dict_rel[(s_idx, raf_i, o_idx)]-1].score_rel += raf_v
            else:
                if s_idx and o_idx:
                    category_id_obj = annotations_det[o_idx].category_id
                    category_id_sub = annotations_det[s_idx].category_id
                    category_id_rel = int(raf_i) + 1
                    score_sub = annotations_det[s_idx].score
                    score_rel = raf_v
                    score_obj = annotations_det[o_idx].score
                    bbox_sub = copy.deepcopy(annotations_det[s_idx].bbox)
                    bbox_obj = copy.deepcopy(annotations_det[o_idx].bbox)
                    ann = AnnotationRaf(self.raf_metas[0].obj_categories,
                                        self.raf_metas[0].rel_categories).set(
                                            category_id_obj, category_id_sub,
                                            category_id_rel, score_sub,
                                            score_rel, score_obj,
                                            bbox_sub, bbox_obj)
                    annotations.append(ann)
                    dict_rel[(s_idx, raf_i, o_idx)] = len(annotations)

        self.occupancy_visualizer.predicted(occupied)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

        return annotations

    # def connection_value(self, ann, raf_scored, start_i, end_i, *, reverse_match=True):
    #     raf_i, forward = self.by_source[start_i][end_i]
    #     raf_f, raf_b = raf_scored.directed(raf_i, forward)
    #     xyv = ann.data[start_i]
    #     xy_scale_s = max(0.0, ann.joint_scales[start_i])
    #
    #     only_max = self.connection_method == 'max'
    #
    #     new_xysv = grow_connection_blend(
    #         raf_f, xyv[0], xyv[1], xy_scale_s, only_max)
    #     keypoint_score = np.sqrt(new_xysv[3] * xyv[2])  # geometric mean
    #     if keypoint_score < self.keypoint_threshold:
    #         return 0.0, 0.0, 0.0, 0.0
    #     if new_xysv[3] == 0.0:
    #         return 0.0, 0.0, 0.0, 0.0
    #     xy_scale_t = max(0.0, new_xysv[2])
    #
    #     # reverse match
    #     if reverse_match:
    #         reverse_xyv = grow_connection_blend(
    #             raf_b, new_xysv[0], new_xysv[1], xy_scale_t, only_max)
    #         if reverse_xyv[2] == 0.0:
    #             return 0.0, 0.0, 0.0, 0.0
    #         if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
    #             return 0.0, 0.0, 0.0, 0.0
    #
    #     return (new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score)
    #
    # def _grow(self, ann, raf_scored, *, reverse_match=True):
    #     frontier = []
    #     in_frontier = set()
    #
    #     def add_to_frontier(start_i):
    #         for end_i, (raf_i, _) in self.by_source[start_i].items():
    #             if ann.data[end_i, 2] > 0.0:
    #                 continue
    #             if (start_i, end_i) in in_frontier:
    #                 continue
    #
    #             max_possible_score = np.sqrt(ann.data[start_i, 2])
    #             if self.confidence_scales is not None:
    #                 max_possible_score *= self.confidence_scales[raf_i]
    #             heapq.heappush(frontier, (-max_possible_score, None, start_i, end_i))
    #             in_frontier.add((start_i, end_i))
    #             ann.frontier_order.append((start_i, end_i))
    #
    #     def frontier_get():
    #         while frontier:
    #             entry = heapq.heappop(frontier)
    #             if entry[1] is not None:
    #                 return entry
    #
    #             _, __, start_i, end_i = entry
    #             if ann.data[end_i, 2] > 0.0:
    #                 continue
    #
    #             new_xysv = self.connection_value(
    #                 ann, raf_scored, start_i, end_i, reverse_match=reverse_match)
    #             if new_xysv[3] == 0.0:
    #                 continue
    #             score = new_xysv[3]
    #             if self.greedy:
    #                 return (-score, new_xysv, start_i, end_i)
    #             if self.confidence_scales is not None:
    #                 raf_i, _ = self.by_source[start_i][end_i]
    #                 score = score * self.confidence_scales[raf_i]
    #             heapq.heappush(frontier, (-score, new_xysv, start_i, end_i))
    #
    #     # seeding the frontier
    #     if ann.score == 0.0:
    #         return
    #     add_to_frontier(ann.category_id-1)
    #     import pdb; pdb.set_trace()
    #     while True:
    #         entry = frontier_get()
    #         if entry is None:
    #             break
    #
    #         _, new_xysv, jsi, jti = entry
    #         if ann.data[jti, 2] > 0.0:
    #             continue
    #
    #         ann.data[jti, :2] = new_xysv[:2]
    #         ann.data[jti, 2] = new_xysv[3]
    #         ann.joint_scales[jti] = new_xysv[2]
    #         ann.decoding_order.append(
    #             (jsi, jti, np.copy(ann.data[jsi]), np.copy(ann.data[jti])))
    #         add_to_frontier(jti)
