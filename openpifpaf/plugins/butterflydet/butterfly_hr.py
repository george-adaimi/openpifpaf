import logging
import time
import numpy as np

from openpifpaf.decoder import utils
from .functional import scalar_square_add_2dgauss, cumulative_average_2d

LOG = logging.getLogger(__name__)

class ButterflyHr(utils.CifHr):
    def fill(self, all_fields, metas):
        start = time.perf_counter()

        if self.accumulated is None:
            shape = (
                all_fields[0].shape[0],
                int((all_fields[0].shape[2]) * metas[0].stride),
                int((all_fields[0].shape[3]) * metas[0].stride),
            )
            ta = np.zeros(shape, dtype=np.float32)
            self.scales_n = np.zeros(shape, dtype=np.float32)
            self.scales_w = np.zeros(shape, dtype="float32")
            self.scales_h = np.zeros(shape, dtype="float32")
            self.widths = np.zeros(shape, dtype="float32")
            self.heights = np.zeros(shape, dtype="float32")
            self.scalew_n = np.zeros(shape, dtype="float32")
            self.scaleh_n = np.zeros(shape, dtype="float32")
            self.width_n = np.zeros(shape, dtype="float32")
            self.height_n = np.zeros(shape, dtype="float32")
        else:
            ta = np.zeros(self.accumulated.shape, dtype=np.float32)

        for meta in metas:
            for t, p, scale_w, scale_h, width, height, n_sw, n_sh, n_w, n_h in zip(ta, all_fields[meta.head_index], self.scales_w, self.scales_h, self.widths, self.heights, self.scalew_n, self.scaleh_n, self.width_n, self.height_n):
                self.accumulate(len(metas), t, scale_w, scale_h, width, height, n_sw, n_sh, n_w, n_h, p, meta.stride, meta.decoder_min_scale)
        ta = np.tanh(ta)
        if self.accumulated is None:
            self.accumulated = ta
        else:
            self.accumulated = np.maximum(ta, self.accumulated)

        LOG.debug('target_intensities %.3fs', time.perf_counter() - start)
        self.debug_visualizer.predicted(self.accumulated)
        return self

    def accumulate(self, len_cifs, t, scale_w, scale_h, width, height, n_sw, n_sh, n_w, n_h, p, stride, min_scale):
        p = p[:, p[0] > self.v_threshold]
        if min_scale:
            p = p[:, p[4] > min_scale / stride]
            p = p[:, p[5] > min_scale / stride]

        v, x, y, w, h, _, __ = p
        x = x * stride
        y = y * stride
        #w = np.exp(w)
        #h = np.exp(h)
        sigma = np.maximum(1.0, 0.1 * np.minimum(w, h) * stride)
        w = w * stride
        h = h * stride
        s_h = np.clip(h/10, a_min=2, a_max=None)
        s_w = np.clip(w/10, a_min=2, a_max=None)
        # Occupancy covers 2sigma.
        # Restrict this accumulation to 1sigma so that seeds for the same joint
        # are properly suppressed.
        if self.fullfields:
            cifdet_nn = np.clip((w/stride)*(h/stride), a_min=16, a_max= None)
            #cifdet_nn = 1
        else:
            cifdet_nn = self.neighbors
        scalar_square_add_2dgauss(
            t, x, y, s_w, s_h, (v / cifdet_nn).astype(np.float32), truncate=0.5)

        cumulative_average_2d(scale_w, n_sw, x, y, s_w, s_h, (s_w), v)
        cumulative_average_2d(scale_h, n_sh, x, y, s_w, s_h, (s_h), v)
        cumulative_average_2d(width, n_w, x, y, s_w, s_h, w, v)
        cumulative_average_2d(height, n_h, x, y, s_w, s_h, h, v)
