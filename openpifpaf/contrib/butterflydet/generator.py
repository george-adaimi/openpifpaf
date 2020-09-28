from collections import defaultdict
import logging
import time
from typing import List

from openpifpaf.decoder.generator import Generator
from openpifpaf.annotation import AnnotationDet
from .butterfly_hr import ButterflyHr
from .butterfly_seeds import ButterflySeeds
from openpifpaf.decoder import nms
from openpifpaf.decoder.occupancy import Occupancy
from openpifpaf import headmeta, visualizer

LOG = logging.getLogger(__name__)


class ButterflyDet(Generator):
    occupancy_visualizer = visualizer.Occupancy()

    def __init__(self, head_metas: List[headmeta.CifDet], *, visualizers=None):
        super().__init__()
        self.metas = head_metas

        self.visualizers = visualizers
        if self.visualizers is None:
            self.visualizers = [visualizer.CifDet(meta) for meta in self.metas]

        self.timers = defaultdict(float)

    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        return [
            ButterflyDet([meta])
            for meta in head_metas
            if isinstance(meta, headmeta.CifDet)
        ]

    def __call__(self, fields):
        start = time.perf_counter()

        if self.visualizers:
            for vis, meta in zip(self.visualizers, self.metas):
                vis.predicted(fields[meta.head_index])

        cifhr = ButterflyHr().fill(fields, self.metas)
        seeds = ButterflySeeds(cifhr).fill(fields, self.metas)
        occupied = Occupancy(cifhr.accumulated.shape, 2, min_scale=2.0)

        annotations = []
        for v, f, x, y, w, h in seeds.get():
            if occupied.get(f, x, y):
                continue
            ann = AnnotationDet(self.metas[0].categories).set(
                f + 1, v, (x - w/2.0, y - h/2.0, w, h))
            annotations.append(ann)
            occupied.set(f, x, y, 0.1 * min(w, h))

        self.occupancy_visualizer.predicted(occupied)

        annotations = nms.Detection().annotations_per_category(annotations, nms_type='snms')
        #annotations = nms.Detection().annotations(annotations)
        # annotations = sorted(annotations, key=lambda a: -a.score)

        LOG.info('annotations %d, decoder = %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
