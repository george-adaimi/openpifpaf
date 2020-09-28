import numpy as np
import functools
import dataclasses

from openpifpaf.encoder.cifdet import CifDetGenerator, CifDet

@functools.lru_cache(maxsize=64)
def create_sink_2d(w, h):
    if w == 1 and h == 1:
        return np.zeros((2, 1, 1))

    sink1d_w = np.linspace((w - 1.0) / 2.0, -(w - 1.0) / 2.0, num=w, dtype=np.float32)
    sink1d_h = np.linspace((h - 1.0) / 2.0, -(h - 1.0) / 2.0, num=h, dtype=np.float32)
    sink = np.stack((
        sink1d_w.reshape(1, -1).repeat(h, axis=0),
        sink1d_h.reshape(-1, 1).repeat(w, axis=1),
    ), axis=0)
    return sink

@dataclasses.dataclass
class FullButterfly(CifDet):
    def __call__(self, image, anns, meta):
        return FullButterflyGenerator(self)(image, anns, meta)

class FullButterflyGenerator(CifDetGenerator):

    def fill_detection(self, f, xy, wh):
        w = np.round(wh[0] + 0.5).astype(np.int)

        h = np.round(wh[1] + 0.5).astype(np.int)
        self.s_offset = [(w-1.0)/2.0, (h-1.0)/2.0]
        ij = np.round(xy - self.s_offset).astype(np.int) + self.config.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + w, miny + h
        if minx + w/2 < 0 or maxx - w/2 > self.intensities.shape[2] or \
           miny + h/2 < 0 or maxy - h/2 > self.intensities.shape[1]:
            return

        offset = xy - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(2, 1, 1)

        self.sink = create_sink_2d(w, h)
        minx_n = max(0, minx)
        miny_n = max(0, miny)
        maxx_n = min(maxx, self.intensities.shape[2])
        maxy_n = min(maxy, self.intensities.shape[1])
        sink = self.sink[:, (miny_n-miny):(miny_n-miny) + (maxy_n-miny_n), (minx_n-minx):(minx_n-minx) + (maxx_n-minx_n)]
        minx = minx_n
        maxx = maxx_n
        miny = miny_n
        maxy = maxy_n

        # mask
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        # core_radius = (self.config.side_length - 1) / 2.0
        # mask_fringe = np.logical_and(
        #     sink_l > core_radius,
        #     sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx],
        # )
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0
        #self.intensities[f, miny:maxy, minx:maxx][mask_fringe] = np.nan

        # update regression
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = sink_reg[:, mask]

        # update wh
        assert wh[0] > 0.0
        assert wh[1] > 0.0
        self.fields_wh[f, :, miny:maxy, minx:maxx][:, mask] = np.expand_dims(wh, 1)

        # update bmin
        self.fields_reg_bmin[f, miny:maxy, minx:maxx][mask] = 1.0
        self.fields_wh_bmin[f, miny:maxy, minx:maxx][mask] = 1.0
