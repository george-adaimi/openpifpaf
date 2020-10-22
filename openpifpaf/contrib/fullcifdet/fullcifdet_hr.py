import logging

import numpy as np

# pylint: disable=import-error
from openpifpaf.functional import scalar_square_add_gauss_with_max
from openpifpaf.decoder import utils
LOG = logging.getLogger(__name__)

class FullCifDetHr(utils.CifHr):
    def accumulate(self, len_cifs, t, p, stride, min_scale):
        p = p[:, p[0] > self.v_threshold]
        if min_scale:
            p = p[:, p[4] > min_scale / stride]
            p = p[:, p[5] > min_scale / stride]

        v, x, y, w, h, _, __ = p
        x = x * stride
        y = y * stride
        sigma = np.maximum(1.0, 0.1 * np.minimum(w, h) * stride)

        # Occupancy covers 2sigma.
        # Restrict this accumulation to 1sigma so that seeds for the same joint
        # are properly suppressed.
        # scalar_square_add_gauss_with_max(
        #     t, x, y, sigma, v / self.neighbors / len_cifs, truncate=1.0)
        scalar_square_add_gauss_with_max(
            t, x, y, sigma, v / len_cifs, truncate=1.0,  max_value=10000.0)
