import dataclasses
import logging
import numpy as np
import torch
from typing import ClassVar

from math import sqrt

from .annrescaler import AnnRescalerRel
from .visualizer import Raf as RafVisualizer
from openpifpaf.utils import create_sink, mask_valid_area
from . import headmeta
LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Raf:
    meta: headmeta.Raf
    rescaler: AnnRescalerRel=None
    v_threshold: int = 0
    visualizer: RafVisualizer = None

    min_size: ClassVar[int] = 3
    fixed_size: ClassVar[bool] = False
    aspect_ratio: ClassVar[float] = 0.0
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta):
        return RafGenerator(self)(image, anns, meta)


class RafGenerator:
    def __init__(self, config: Raf):
        self.config = config
        # self.skeleton_m1 = np.asarray(config.skeleton) - 1
        # self.sparse_skeleton_m1 = (
        #     np.asarray(config.sparse_skeleton) - 1
        #     if config.sparse_skeleton else None)
        self.visualizer = config.visualizer or RafVisualizer(config.meta)
        self.rescaler = config.rescaler or AnnRescalerRel(
            config.meta.stride, len(config.meta.rel_categories))

        if self.config.fixed_size:
            assert self.config.aspect_ratio == 0.0

        LOG.debug('only_in_field_of_view = %s, paf min size = %d',
                  config.meta.only_in_field_of_view,
                  self.config.min_size)

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_bmin1 = None
        self.fields_bmin2 = None
        #self.fields_scale1 = None
        #self.fields_scale2 = None
        self.fields_reg_l = None

    def __call__(self, image, anns, meta):
        #self.meta = meta
        width_height_original = image.shape[2:0:-1]
        detections = self.rescaler.relations(anns)
        bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                        crowd_margin=(self.config.min_size - 1) / 2)

        valid_area = self.rescaler.valid_area(meta)
        LOG.debug('valid area: %s', valid_area)

        self.init_fields(len(self.config.meta.rel_categories), bg_mask)
        self.fill(detections)
        fields = self.fields(valid_area)

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns)

        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[2] + 2 * self.config.padding
        field_h = bg_mask.shape[1] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg1 = np.full((n_fields, 6, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg2 = np.full((n_fields, 6, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg1[:, 2:] = np.inf
        self.fields_reg2[:, 2:] = np.inf
        self.fields_bmin1 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_bmin2 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        #self.fields_scale1 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        #self.fields_scale2 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][bg_mask == 0] = np.nan

    def fill(self, detections):
        for det_index, (k, ann) in enumerate(detections.items()):
            if not len(ann['object_index']) > 0:
                continue
            bbox = ann['bbox']
            det_id = ann['detection_id']
            category_id = ann['category_id']
            for object_id, predicate in zip(ann['object_index'], ann['predicate']):
                bbox_object = detections[object_id]['bbox']
                object_category = detections[object_id]['category_id']
                self.fill_keypoints(predicate, bbox, bbox_object, [other_ann['bbox'] for other_ann in detections.values() if (other_ann['detection_id'] != det_id and other_ann['category_id']==category_id)], [other_ann['bbox']  for other_ann in detections.values() if (other_ann['detection_id'] != object_id and other_ann['category_id']==object_category)])

    def fill_keypoints(self, predicate, subject_det, object_det, other_subj, other_obj):
        #scale = self.config.rescaler.scale(keypoints)
        scale1 = 0.1 * np.minimum(subject_det[2], subject_det[3])
        scale2 = 0.1 * np.minimum(object_det[2],object_det[3])
        joint1 = subject_det[:2] + 0.5 * subject_det[2:]
        joint2 = object_det[:2] + 0.5 * object_det[2:]
        #for paf_i, (joint1i, joint2i) in enumerate(self.skeleton_m1):
        # joint1 = keypoints[joint1i]
        # joint2 = keypoints[joint2i]
        # if joint1[2] <= self.config.v_threshold or joint2[2] <= self.config.v_threshold:
        #     continue

        # check if there are shorter connections in the sparse skeleton
        # if self.sparse_skeleton_m1 is not None:
        #     d = np.linalg.norm(joint1[:2] - joint2[:2]) / self.config.dense_to_sparse_radius
        #     if self.shortest_sparse(joint1i, keypoints) < d \
        #        and self.shortest_sparse(joint2i, keypoints) < d:
        #         continue

        # if there is no continuous visual connection, endpoints outside
        # the field of view cannot be inferred
        if self.config.meta.only_in_field_of_view:
            # LOG.debug('fov check: j1 = %s, j2 = %s', joint1, joint2)
            if joint1[0] < 0 or \
               joint2[0] < 0 or \
               joint1[0] > self.intensities.shape[2] - 1 - 2 * self.config.padding or \
               joint2[0] > self.intensities.shape[2] - 1 - 2 * self.config.padding:
               return
            if joint1[1] < 0 or \
               joint2[1] < 0 or \
               joint1[1] > self.intensities.shape[1] - 1 - 2 * self.config.padding or \
               joint2[1] > self.intensities.shape[1] - 1 - 2 * self.config.padding:
               return

        # other_j1s = [other_kps[joint1i] for other_kps in other_keypoints
        #              if other_kps[joint1i, 2] > self.config.v_threshold]
        # other_j2s = [other_kps[joint2i] for other_kps in other_keypoints
        #              if other_kps[joint2i, 2] > self.config.v_threshold]
        max_r1 = RafGenerator.max_r(subject_det, other_subj)
        max_r2 = RafGenerator.max_r(object_det, other_obj)

        if self.config.meta.sigmas is None:
            scale1, scale2 = scale1, scale2
        else:
            scale1 = scale1 * self.config.meta.sigmas[joint1i]
            scale2 = scale2 * self.config.meta.sigmas[joint2i]
        #scale1 = np.min([scale1, np.min(max_r1) * 0.25])
        #scale2 = np.min([scale2, np.min(max_r2) * 0.25])
        self.fill_association(predicate, joint1, joint2, scale1, scale2, max_r1, max_r2)

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

    def fill_association(self, paf_i, joint1, joint2, scale1, scale2, max_r1, max_r2):
        # offset between joints
        offset = joint2[:2] - joint1[:2]
        offset_d = np.linalg.norm(offset)

        # dynamically create s
        s = max(self.config.min_size, int(offset_d * self.config.aspect_ratio))
        # s = max(s, min(int(scale1), int(scale2)))
        sink = create_sink(s)
        s_offset = (s - 1.0) / 2.0

        # set fields
        num = max(2, int(np.ceil(offset_d)))
        fmargin = (s_offset + 1) / (offset_d + np.spacing(1))
        fmargin = np.clip(fmargin, 0.25, 0.4)
        # fmargin = 0.0
        frange = np.linspace(fmargin, 1.0-fmargin, num=num)
        if self.config.fixed_size:
            frange = [0.5]
        for f in frange:
            fij = np.round(joint1[:2] + f * offset - s_offset) + self.config.padding
            fminx, fminy = int(fij[0]), int(fij[1])
            fmaxx, fmaxy = fminx + s, fminy + s
            if fminx < 0 or fmaxx > self.intensities.shape[2] or \
               fminy < 0 or fmaxy > self.intensities.shape[1]:
                continue
            fxy = fij - self.config.padding + s_offset

            # precise floating point offset of sinks
            joint1_offset = (joint1[:2] - fxy).reshape(2, 1, 1)
            joint2_offset = (joint2[:2] - fxy).reshape(2, 1, 1)
            sink1 = sink + joint1_offset
            sink2 = sink + joint2_offset

            # mask
            # perpendicular distance computation:
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            # Coordinate systems for this computation is such that
            # joint1 is at (0, 0).
            sink_l = np.fabs(
                offset[1] * sink1[0]
                - offset[0] * sink1[1]
            ) / (offset_d + 0.01)
            mask = sink_l < self.fields_reg_l[paf_i, fminy:fmaxy, fminx:fmaxx]
            mask_peak = np.logical_and(mask, np.linalg.norm(sink1 + sink2, axis=0) < 0.7)
            self.fields_reg_l[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

            # update intensity
            self.intensities[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = 1.0
            self.intensities[paf_i, fminy:fmaxy, fminx:fmaxx][mask_peak] = 1.0

            # update regressions
            patch1 = self.fields_reg1[paf_i, :, fminy:fmaxy, fminx:fmaxx]
            patch1[:2, mask] = sink1[:, mask]
            patch1[2:, mask] = np.expand_dims(max_r1, 1) * 0.5
            patch2 = self.fields_reg2[paf_i, :, fminy:fmaxy, fminx:fmaxx]
            patch2[:2, mask] = sink2[:, mask]
            patch2[2:, mask] = np.expand_dims(max_r2, 1) * 0.5

            # update bmin
            self.fields_bmin1[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = 1.0
            self.fields_bmin2[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = 1.0

            # update scale
            #assert np.isnan(scale1) or  0.0 < scale1 < 100.0
            #self.fields_scale1[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = scale1
            #assert np.isnan(scale2) or  0.0 < scale2 < 100.0
            #self.fields_scale2[paf_i, fminy:fmaxy, fminx:fmaxx][mask] = scale2

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg1 = self.fields_reg1[:, :, p:-p, p:-p]
        fields_reg2 = self.fields_reg2[:, :, p:-p, p:-p]
        fields_bmin1 = self.fields_bmin1[:, p:-p, p:-p]
        fields_bmin2 = self.fields_bmin2[:, p:-p, p:-p]
        #fields_scale1 = self.fields_scale1[:, p:-p, p:-p]
        #fields_scale2 = self.fields_scale2[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg1[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg1[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin2, valid_area, fill_value=np.nan)
        #mask_valid_area(fields_scale1, valid_area, fill_value=np.nan)
        #mask_valid_area(fields_scale2, valid_area, fill_value=np.nan)

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg1[:, :2],  # TODO dropped margin components for now
            fields_reg2[:, :2],
            np.expand_dims(fields_bmin1, 1),
            np.expand_dims(fields_bmin2, 1),
            #np.expand_dims(fields_scale1, 1),
            #np.expand_dims(fields_scale2, 1),
        ], axis=1))
