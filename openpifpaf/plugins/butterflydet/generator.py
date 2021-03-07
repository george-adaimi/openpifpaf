import argparse
from collections import defaultdict
import logging
import time
from typing import List

from openpifpaf.decoder import Decoder
from openpifpaf.annotation import AnnotationDet
from .butterfly_hr import ButterflyHr
from .butterfly_seeds import ButterflySeeds
from openpifpaf.decoder import utils
from openpifpaf import headmeta, visualizer

LOG = logging.getLogger(__name__)


class ButterflyDet(Decoder):
    occupancy_visualizer = visualizer.Occupancy()
    fullfields = False
    def __init__(self, head_metas: List[headmeta.CifDet], *, visualizers=None):
        super().__init__()
        self.metas = head_metas

        self.visualizers = visualizers
        if self.visualizers is None:
            self.visualizers = [visualizer.CifDet(meta) for meta in self.metas]

        self.timers = defaultdict(float)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Commond line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('ButterflyDet decoder')

        group.add_argument('--butterflydet-fullfields',
                           default=False, action='store_true',
                           help='greedy decoding')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""

        cls.fullfields = args.butterflydet_fullfields

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

        cifhr = ButterflyHr(self.fullfields).fill(fields, self.metas)
        seeds = ButterflySeeds(cifhr).fill(fields, self.metas)
        occupied = utils.Occupancy(cifhr.accumulated.shape, 2, min_scale=2.0)

        annotations = []
        for v, f, x, y, w, h in seeds.get():
            if occupied.get(f, x, y):
                continue
            ann = AnnotationDet(self.metas[0].categories).set(
                f + 1, v, (x - w/2.0, y - h/2.0, w, h))
            annotations.append(ann)
            occupied.set(f, x, y, 0.1 * min(w, h))

        self.occupancy_visualizer.predicted(occupied)

        annotations = utils.nms.Detection().annotations_per_category(annotations, nms_type='snms')
        #annotations = utils.nms.Detection().annotations(annotations)
        # annotations = sorted(annotations, key=lambda a: -a.score)

        LOG.info('annotations %d, decoder = %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
