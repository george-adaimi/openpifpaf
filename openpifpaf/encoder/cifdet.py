import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch

from .annrescaler import AnnRescaler
from ..visualizer import CifDet as CifDetVisualizer
from ..utils import create_sink, mask_valid_area, create_sink_2d

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class CifDet:
    n_categories: int
    rescaler: AnnRescaler
    v_threshold: int = 0

    visualizer: CifDetVisualizer = None
    ignore_regions: bool = False
    full_fields: bool = False

    side_length: ClassVar[int] = 5
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta):
        return CifDetGenerator(self)(image, anns, meta)


class CifDetGenerator(object):
    def __init__(self, config: CifDet):
        self.config = config

        self.intensities = None
        self.fields_reg = None
        self.fields_wh = None
        self.fields_reg_l = None

        self.sink = create_sink(config.side_length)
        self.s_offset = (config.side_length - 1.0) / 2.0
        self.fill_fn = self.fill_detection
        if config.ignore_regions:
            self.fill_fn = self.fill_detection_with_ignore_regions
        elif config.full_fields:
            self.fill_fn = self.fill_detection_full

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

        detections = self.config.rescaler.detections(anns)
        bg_mask = self.config.rescaler.bg_mask(anns, width_height_original,
                                               crowd_margin=(self.config.side_length - 1) / 2)
        valid_area = self.config.rescaler.valid_area(meta)
        LOG.debug('valid area: %s, pif side length = %d', valid_area, self.config.side_length)

        n_fields = self.config.n_categories
        self.init_fields(n_fields, bg_mask)
        self.fill(detections)
        fields = self.fields(valid_area)

        self.config.visualizer.processed_image(image, meta)
        self.config.visualizer.targets(fields, detections=detections)

        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[-1] + 2 * self.config.padding
        field_h = bg_mask.shape[-2] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_wh = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][bg_mask == 0] = np.nan

    def fill(self, detections):
        for category_id, bbox in detections:
            xy = bbox[:2] + 0.5 * bbox[2:]
            wh = bbox[2:]
            self.fill_fn(category_id - 1, xy, wh)

    @staticmethod
    def quadrant(xys):
        q = np.zeros((xys.shape[0],), dtype=np.int)
        q[xys[:, 0] < 0.0] += 1
        q[xys[:, 1] < 0.0] += 2
        return q

    @classmethod
    def max_r(cls, xyv, other_xyv):
        out = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        if not other_xyv:
            return out

        other_xyv = np.asarray(other_xyv)
        diffs = other_xyv[:, :2] - np.expand_dims(xyv[:2], 0)
        qs = cls.quadrant(diffs)
        for q in range(4):
            if not np.any(qs == q):
                continue
            out[q] = np.min(np.linalg.norm(diffs[qs == q], axis=1))

        return out

    def fill_detection(self, f, xy, wh):
        ij = np.round(xy - self.s_offset).astype(np.int) + self.config.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.config.side_length, miny + self.config.side_length
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return

        offset = xy - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(2, 1, 1)

        # mask
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        core_radius = (self.config.side_length - 1) / 2.0
        mask_fringe = np.logical_and(
            sink_l > core_radius,
            sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx],
        )
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0
        self.intensities[f, miny:maxy, minx:maxx][mask_fringe] = np.nan

        # update regression
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = sink_reg[:, mask]

        # update wh
        assert wh[0] > 0.0
        assert wh[1] > 0.0
        self.fields_wh[f, :, miny:maxy, minx:maxx][:, mask] = np.expand_dims(wh, 1)

    def fill_detection_full(self, f, xy, wh):
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
        sink_reg = sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)

        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]

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

    def fill_detection_with_ignore_regions(self, f, xy, wh):
        ij = np.round(xy - self.s_offset).astype(np.int) + self.config.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.config.side_length, miny + self.config.side_length
        if minx + 2 < 0 or maxx - 2 > self.intensities.shape[2] or \
           miny + 2 < 0 or maxy - 2 > self.intensities.shape[1]:
            return

        offset = xy - (ij + self.s_offset - self.config.padding)
        if wh[0] > 16 or wh[1]>16:
            sigma_ignore, sigma_eff = 0.5, 0.2
            w_ignore, h_ignore = np.round(sigma_ignore*wh[0] + 0.5).astype(np.int), np.round(sigma_ignore*wh[1]+ 0.5).astype(np.int)
            xy_offset = [(w_ignore - 1.0) / 2.0, (h_ignore - 1.0) / 2.0]
            ij = np.round(xy - xy_offset).astype(np.int) + self.config.padding
            if wh[0]>16:
                minx_ignore = int(ij[0])
            else:
                minx_ignore = minx
                w_ignore = 4
            if wh[1]>16:
                miny_ignore = int(ij[1])
            else:
                miny_ignore = miny
                h_ignore = 4
            maxx_ignore, maxy_ignore = minx_ignore + w_ignore, miny_ignore + h_ignore
            self.intensities[f, miny_ignore:maxy_ignore, minx_ignore:maxx_ignore] = np.nan
        #self.sink = create_sink_2d(w, h)
        minx_n = max(0, minx)
        miny_n = max(0, miny)
        maxx_n = min(maxx, self.intensities.shape[2])
        maxy_n = min(maxy, self.intensities.shape[1])
        sink = self.sink[:, (miny_n-miny):(miny_n-miny) + (maxy_n-miny_n), (minx_n-minx):(minx_n-minx) + (maxx_n-minx_n)]
        minx = minx_n
        maxx = maxx_n
        miny = miny_n
        maxy = maxy_n

        offset = offset.reshape(2, 1, 1)

        # mask
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]

        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0

        # update regression
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = sink_reg[:, mask]

        # update wh
        assert wh[0] > 0.0
        assert wh[1] > 0.0
        self.fields_wh[f, :, miny:maxy, minx:maxx][:, mask] = np.expand_dims(wh, 1)

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg = self.fields_reg[:, :, p:-p, p:-p]
        fields_wh = self.fields_wh[:, :, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_wh[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_wh[:, 1], valid_area, fill_value=np.nan)

        return (
            torch.from_numpy(intensities),
            torch.from_numpy(fields_reg),
            torch.from_numpy(fields_wh),
        )
