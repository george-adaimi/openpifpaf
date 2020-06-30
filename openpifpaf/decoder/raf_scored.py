import logging
import time

import numpy as np

# pylint: disable=import-error
from ..functional import scalar_values_3d
from .field_config import FieldConfig

LOG = logging.getLogger(__name__)

class RafScored:
    default_score_th = 0.1

    def __init__(self, cifhr, config: FieldConfig, *, score_th=None, cif_floor=0.1):
        self.cifhr = cifhr
        self.config = config
        self.score_th = score_th or self.default_score_th
        self.cif_floor = cif_floor

        self.forward = None
        self.backward = None

    def directed(self, caf_i, forward):
        if forward:
            return self.forward[caf_i], self.backward[caf_i]

        return self.backward[caf_i], self.forward[caf_i]

    def fill_raf(self, raf, stride, min_distance=0.0, max_distance=None):
        start = time.perf_counter()

        if self.forward is None:
            self.forward = [np.empty((9, 0), dtype=raf.dtype) for _ in raf]
            self.backward = [np.empty((9, 0), dtype=raf.dtype) for _ in raf]

        import pdb; pdb.set_trace()
        for raf_i, nine in enumerate(raf):
            assert nine.shape[0] == 9
            import pdb; pdb.set_trace()
            mask = nine[0] > self.score_th
            if not np.any(mask):
                continue
            nine = nine[:, mask]

            if min_distance:
                dist = np.linalg.norm(nine[1:3] - nine[5:7], axis=0)
                mask_dist = dist > min_distance / stride
                nine = nine[:, mask_dist]

            if max_distance:
                dist = np.linalg.norm(nine[1:3] - nine[5:7], axis=0)
                mask_dist = dist < max_distance / stride
                nine = nine[:, mask_dist]

            nine = np.copy(nine)
            nine[(1, 2, 3, 4, 5, 6, 7, 8), :] *= stride
            scores = nine[0]

            cifhr_values = scalar_values_3d(self.cifhr, nine[1], nine[2], default=0.0)
            cifhr_s = np.max(cifhr_values, axis=0)
            index_s = np.amax(cifhr_values, axis=0)
            scores_s = scores * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_s)
            mask_s = scores_s > self.score_th
            d9_s = np.copy(nine[:, mask_s][(0, 5, 6, 7, 8, 1, 2, 3, 4), :])
            d9_s[0] = scores_s[mask_s]
            import pdb; pdb.set_trace()
            self.backward[raf_i] = np.concatenate((self.backward[raf_i], d9_s), axis=1)

            cifhr_values = scalar_values_3d(self.cifhr, nine[5], nine[6], default=0.0)
            cifhr_o = np.max(cifhr_values, axis=0)
            index_o = np.amax(cifhr_values, axis=0)
            scores_o = scores * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_o)
            mask_o = scores_o > self.score_th
            d9_o = np.copy(nine[:, mask_o])
            d9_o[0] = scores_o[mask_o]
            self.forward[raf_i] = np.concatenate((self.forward[raf_i], d9_o), axis=1)
        import pdb; pdb.set_trace()
        LOG.debug('scored caf (%d, %d) in %.3fs',
                  sum(f.shape[1] for f in self.forward),
                  sum(b.shape[1] for b in self.backward),
                  time.perf_counter() - start)
        return self

    def fill(self, fields):
        for raf_i, stride, min_distance, max_distance in zip(
                self.config.caf_indices,
                self.config.caf_strides,
                self.config.caf_min_distances,
                self.config.caf_max_distances):
            self.fill_raf(fields[raf_i], stride,
                          min_distance=min_distance, max_distance=max_distance)

        return self
