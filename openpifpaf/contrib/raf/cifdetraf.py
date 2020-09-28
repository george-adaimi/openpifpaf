import argparse
from collections import defaultdict
import heapq
import logging
import time
from typing import List

import heapq
import numpy as np

from openpifpaf.annotation import AnnotationDet
from openpifpaf.decoder.cif_hr import CifDetHr
from openpifpaf.decoder.cif_seeds import CifDetSeeds
from .raf_scored import RafScored
from openpifpaf.decoder.generator import Generator
from openpifpaf.decoder import nms as nms_module
from openpifpaf.decoder.occupancy import Occupancy
from .headmeta import Raf
from openpifpaf import headmeta,visualizer
from . import visualizer as visualizer_raf
from openpifpaf.decoder.generator import Generator

LOG = logging.getLogger(__name__)

class CifDetRaf(Generator):
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
                raf_metas: List[Raf],
                cifdet_metas: List[headmeta.CifDet],
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
            self.nms = nms_module.Detection()


        self.confidence_scales = raf_metas[0].decoder_confidence_scales

        self.timers = defaultdict(float)

        # init by_target and by_source
        self.by_target = defaultdict(dict)

        self.by_source = defaultdict(dict)

    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        return [
            CifDetRaf([meta], [meta_next])
            for meta, meta_next in zip(head_metas[:-1], head_metas[1:])
            if (isinstance(meta, Raf)
                and isinstance(meta_next, headmeta.CifDet))
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

        cifdethr = CifDetHr().fill(fields, self.cifdet_metas)
        seeds = CifDetSeeds(cifdethr.accumulated).fill(fields, self.cifdet_metas)

        raf_scored = RafScored(cifdethr.accumulated).fill(fields, self.raf_metas)

        occupied = Occupancy(cifdethr.accumulated.shape, 2, min_scale=4)
        annotations = []


        def mark_occupied(ann):
            for joint_i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue

                width = ann.joint_scales[joint_i]
                occupied.set(joint_i, xyv[0], xyv[1], width)  # width = 2 * sigma

        for v, f, x, y, w, h in seeds.get():
            if occupied.get(f, x, y):
                continue
            ann = AnnotationDet(self.metas[0].categories).set(f + 1, v, (x - w/2.0, y - h/2.0, w, h))
            self._grow(ann, raf_scored)
            annotations.append(ann)
            #mark_occupied(ann)
            occupied.set(f, x, y, 0.1 * min(w, h))

        self.occupancy_visualizer.predicted(occupied)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

        if self.nms is not None:
            annotations = self.nms.annotations(annotations)

        LOG.info('%d annotations: %s', len(annotations),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        return annotations

    def connection_value(self, ann, raf_scored, start_i, end_i, *, reverse_match=True):
        raf_i, forward = self.by_source[start_i][end_i]
        raf_f, raf_b = raf_scored.directed(raf_i, forward)
        xyv = ann.data[start_i]
        xy_scale_s = max(0.0, ann.joint_scales[start_i])

        only_max = self.connection_method == 'max'

        new_xysv = grow_connection_blend(
            raf_f, xyv[0], xyv[1], xy_scale_s, only_max)
        keypoint_score = np.sqrt(new_xysv[3] * xyv[2])  # geometric mean
        if keypoint_score < self.keypoint_threshold:
            return 0.0, 0.0, 0.0, 0.0
        if new_xysv[3] == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        xy_scale_t = max(0.0, new_xysv[2])

        # reverse match
        if reverse_match:
            reverse_xyv = grow_connection_blend(
                raf_b, new_xysv[0], new_xysv[1], xy_scale_t, only_max)
            if reverse_xyv[2] == 0.0:
                return 0.0, 0.0, 0.0, 0.0
            if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
                return 0.0, 0.0, 0.0, 0.0

        return (new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score)

    def _grow(self, ann, raf_scored, *, reverse_match=True):
        frontier = []
        in_frontier = set()

        def add_to_frontier(start_i):
            for end_i, (raf_i, _) in self.by_source[start_i].items():
                if ann.data[end_i, 2] > 0.0:
                    continue
                if (start_i, end_i) in in_frontier:
                    continue

                max_possible_score = np.sqrt(ann.data[start_i, 2])
                if self.confidence_scales is not None:
                    max_possible_score *= self.confidence_scales[raf_i]
                heapq.heappush(frontier, (-max_possible_score, None, start_i, end_i))
                in_frontier.add((start_i, end_i))
                ann.frontier_order.append((start_i, end_i))

        def frontier_get():
            while frontier:
                entry = heapq.heappop(frontier)
                if entry[1] is not None:
                    return entry

                _, __, start_i, end_i = entry
                if ann.data[end_i, 2] > 0.0:
                    continue

                new_xysv = self.connection_value(
                    ann, raf_scored, start_i, end_i, reverse_match=reverse_match)
                if new_xysv[3] == 0.0:
                    continue
                score = new_xysv[3]
                if self.greedy:
                    return (-score, new_xysv, start_i, end_i)
                if self.confidence_scales is not None:
                    raf_i, _ = self.by_source[start_i][end_i]
                    score = score * self.confidence_scales[raf_i]
                heapq.heappush(frontier, (-score, new_xysv, start_i, end_i))

        # seeding the frontier
        if ann.score == 0.0:
            return
        add_to_frontier(ann.field_i)

        while True:
            entry = frontier_get()
            if entry is None:
                break

            _, new_xysv, jsi, jti = entry
            if ann.data[jti, 2] > 0.0:
                continue

            ann.data[jti, :2] = new_xysv[:2]
            ann.data[jti, 2] = new_xysv[3]
            ann.joint_scales[jti] = new_xysv[2]
            ann.decoding_order.append(
                (jsi, jti, np.copy(ann.data[jsi]), np.copy(ann.data[jti])))
            add_to_frontier(jti)
