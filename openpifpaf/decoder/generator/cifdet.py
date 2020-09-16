from collections import defaultdict
import logging
import time

from .generator import Generator
from ...annotation import AnnotationDet
from ..field_config import FieldConfig
from ..cif_hr import CifDetHr, ButterflyHr
from ..cif_seeds import CifDetSeeds, ButterflySeeds, FullButterfly
from ..butterfly_og import Butterfly
from .. import nms
from ..occupancy import Occupancy
from ... import visualizer

LOG = logging.getLogger(__name__)


class CifDet(Generator):
    occupancy_visualizer = visualizer.Occupancy()

    def __init__(self, field_config: FieldConfig, categories, *, worker_pool=None):
        super().__init__(worker_pool)
        self.field_config = field_config
        self.categories = categories

        self.timers = defaultdict(float)

    def __call__(self, fields):
        start = time.perf_counter()

        if self.field_config.cif_visualizers:
            for vis, cif_i in zip(self.field_config.cif_visualizers, self.field_config.cif_indices):
                vis.predicted(fields[cif_i])

        if True:
            fullButterfly = False
            if not fullButterfly:
                #cifhr = CifDetHr(self.field_config).fill(fields)
                #seeds = CifDetSeeds(cifhr, self.field_config).fill(fields)
                cifhr = ButterflyHr(self.field_config).fill(fields)
                seeds = ButterflySeeds(cifhr, self.field_config).fill(fields)
                occupied = Occupancy(cifhr.accumulated.shape, 2, min_scale=2.0)

                annotations = []
                for v, f, x, y, w, h in seeds.get():
                    if occupied.get(f, x, y):
                        continue
                    ann = AnnotationDet(self.categories).set(f, v, (x - w/2.0, y - h/2.0, w, h))
                    annotations.append(ann)
                    occupied.set(f, x, y, 0.1 * min(w, h))

                self.occupancy_visualizer.predicted(occupied)
            else:
                seeds = FullButterfly(None, self.field_config).fill(fields)

                annotations = []
                for v, f, x, y, w, h in seeds.get():
                    ann = AnnotationDet(self.categories).set(f, v, (x - w/2.0, y - h/2.0, w, h))
                    annotations.append(ann)

        else:
            annotations = Butterfly(stride=self.field_config.cif_strides[0], seed_threshold=CifDetSeeds.threshold,
                         head_index=self.field_config.cif_indices[0],
                         head_names='cifdet',
                         profile=None,
                         debug_visualizer=self.field_config.cif_visualizers[0])(fields)

        annotations = nms.Detection().annotations_fixed(annotations, nms_type='nms')
        annotations = nms.Detection().annotations(annotations, iou_threshold=0.9,nms_type='nms')
        # annotations = sorted(annotations, key=lambda a: -a.score)

        LOG.info('annotations %d, decoder = %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
