import logging
import time

from typing import List

import numpy as np

# pylint: disable=import-error
from .functional import scalar_values_3d
from . import headmeta

LOG = logging.getLogger(__name__)

class RafAnalyzer:
    default_score_th = 0.2

    def __init__(self, cifhr, *, score_th=None, cif_floor=0.1):
        self.cifhr = cifhr
        self.score_th = score_th or self.default_score_th
        self.cif_floor = cif_floor

        #self.triplets = defaultdict([])
        self.triplets = np.empty((0,8))

    def fill_single(self, all_fields, meta: headmeta.Raf):
        start = time.perf_counter()
        raf = all_fields[meta.head_index]

        for raf_i, nine in enumerate(raf):
            mask = nine[0] > self.score_th
            if not np.any(mask):
                continue
            nine = nine[:, mask]
            if nine.shape[0] == 9:
                nine[(1, 2, 3, 4, 5, 6, 7, 8), :] *= meta.stride
            else:
                nine[(1, 2, 3, 4, 5, 6), :] *= meta.stride

            cifhr_values = scalar_values_3d(self.cifhr, nine[1], nine[2], default=0.0)
            cifhr_s = np.max(cifhr_values, axis=0)
            index_s = np.amax(cifhr_values, axis=0)
            nine[0] = nine[0] * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_s)

            cifhr_values = scalar_values_3d(self.cifhr, nine[3], nine[4], default=0.0)
            cifhr_o = np.max(cifhr_values, axis=0)
            index_o = np.amax(cifhr_values, axis=0)
            nine[0] = nine[0] * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_o)
            #self.triplets[index_s].append([nine[0], index_s, nine[1], nine[2], raf_i, index_o, nine[3], nine[4], False])
            self.triplets = np.concatenate((self.triplets, np.column_stack([nine[0], index_s, nine[1], nine[2], [raf_i]*nine[0].shape[0], index_o, nine[3], nine[4]])))

        return self



    def fill(self, all_fields, metas: List[headmeta.Raf]):
        for meta in metas:
            self.fill_single(all_fields, meta)

        return self
