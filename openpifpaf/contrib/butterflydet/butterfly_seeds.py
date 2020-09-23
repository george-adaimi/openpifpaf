import logging
import time

from openpifpaf.functional import scalar_values
from openpifpaf.decoder import CifSeeds

LOG = logging.getLogger(__name__)

class ButterflySeeds(CifSeeds):
    def fill_single(self, all_fields, meta: headmeta.CifDet):
        start = time.perf_counter()

        cif = all_fields[meta.head_index]
        for field_i, p in enumerate(cif):
            p = p[:, p[0] > self.threshold/2.0]
            #p = p[:, p[0] > self.threshold]
            if meta.decoder_min_scale:
                p = p[:, p[4] > meta.decoder_min_scale / meta.stride]
                p = p[:, p[5] > meta.decoder_min_scale / meta.stride]
            c, x, y, w, h, _, __ = p
            v = scalar_values(self.cifhr.accumulated[field_i], x * meta.stride, y * meta.stride, default=0.0)
            w = scalar_values(self.cifhr.widths[field_i], x * meta.stride, y * meta.stride)
            h = scalar_values(self.cifhr.heights[field_i], x * meta.stride, y * meta.stride)
            v = 0.9 * v + 0.1 * c
            if self.score_scale != 1.0:
                v = v * self.score_scale
            m = v > self.threshold
            x = x[m] * meta.stride
            y = y[m] * meta.stride
            v = v[m]
            w = w[m]
            h = h[m]

            for vv, xx, yy, ww, hh in zip(v, x, y, w, h):
                self.seeds.append((vv, field_i, xx, yy, ww, hh))

        LOG.debug('seeds %d, %.3fs', len(self.seeds), time.perf_counter() - start)
        return self
