from collections import defaultdict
import logging
import time

from ...annotation import AnnotationDet
from ..field_config import FieldConfig
from ..cif_hr import CifDetHr
from ..cif_seeds import CifDetSeeds
from ..raf_scored import RafScored
from .generator import Generator
from .. import nms as nms_module
from ..occupancy import Occupancy
from ... import visualizer

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

    def __init__(self, field_config: FieldConfig,
                 categories,
                 *,
                 confidence_scales=None,
                 worker_pool=None,
                 nms=True):

        super().__init__(worker_pool)
        if nms is True:
            nms = nms_module.Detection()

        self.field_config = field_config

        self.confidence_scales = confidence_scales
        self.nms = nms

        self.timers = defaultdict(float)

        # init by_target and by_source
        self.by_target = defaultdict(dict)

    def __call__(self, fields, initial_annotations=None, meta=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        import pdb; pdb.set_trace()
        if self.field_config.cif_visualizers:
            for vis, cif_i in zip(self.field_config.cif_visualizers, self.field_config.cif_indices):
                vis.predicted(fields[cif_i])
        if self.field_config.caf_visualizers:
            for vis, caf_i in zip(self.field_config.caf_visualizers, self.field_config.caf_indices):
                vis.predicted(fields[caf_i])

        cifhr = CifDetHr(self.field_config).fill(fields)
        seeds = CifDetSeeds(cifhr, self.field_config).fill(fields)

        import pdb; pdb.set_trace()
        raf_scored = RafScored(cifhr.accumulated, self.field_config).fill(fields)

        occupied = Occupancy(cifhr.accumulated.shape, 2, min_scale=4)
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
            ann = AnnotationDet(self.categories).set(f, v, (x - w/2.0, y - h/2.0, w, h))
            self._grow(ann, raf_scored)
            annotations.append(ann)
            occupied.set(f, x, y, 0.1 * min(w, h))

        self.occupancy_visualizer.predicted(occupied)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

        if self.nms is not None:
            annotations = self.nms.annotations(annotations)

        LOG.info('%d annotations: %s', len(annotations),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])
        return annotations

    def _grow(self, ann, raf_scored, *, reverse_match=True):
        frontier = PriorityQueue()
        in_frontier = set()

        def add_to_frontier(start_i):
            for end_i, (caf_i, _) in self.by_source[start_i].items():
                if ann.data[end_i, 2] > 0.0:
                    continue
                if (start_i, end_i) in in_frontier:
                    continue

                max_possible_score = np.sqrt(ann.data[start_i, 2])
                if self.confidence_scales is not None:
                    max_possible_score *= self.confidence_scales[caf_i]
                frontier.put((-max_possible_score, None, start_i, end_i))
                in_frontier.add((start_i, end_i))
                ann.frontier_order.append((start_i, end_i))

        def frontier_get():
            while frontier.qsize():
                entry = frontier.get()
                if entry[1] is not None:
                    return entry

                _, __, start_i, end_i = entry
                if ann.data[end_i, 2] > 0.0:
                    continue

                new_xysv = self.connection_value(
                    ann, caf_scored, start_i, end_i, reverse_match=reverse_match)
                if new_xysv[3] == 0.0:
                    continue
                score = new_xysv[3]
                if self.greedy:
                    return (-score, new_xysv, start_i, end_i)
                if self.confidence_scales is not None:
                    caf_i, _ = self.by_source[start_i][end_i]
                    score *= self.confidence_scales[caf_i]
                frontier.put((-score, new_xysv, start_i, end_i))

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
