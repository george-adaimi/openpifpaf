import copy
import logging
import numpy as np
import os

from .base import BaseVisualizer
from ..annotation import Annotation
from .. import show

try:
    import matplotlib.cm
    CMAP_BLUES_NAN = copy.copy(matplotlib.cm.get_cmap('Blues')).set_bad('white', alpha=0.5)
except ImportError:
    CMAP_BLUES_NAN = None

LOG = logging.getLogger(__name__)


class Caf(BaseVisualizer):
    show_margin = False
    show_background = False
    show_confidences = False
    show_regressions = False
    fig_file = None

    def __init__(self, head_name, *, stride=1, keypoints=None, skeleton=None):
        super().__init__(head_name)

        self.stride = stride
        self.keypoints = keypoints
        self.skeleton = skeleton

        self.keypoint_painter = show.KeypointPainter(xy_scale=self.stride)

    def targets(self, field, keypoint_sets):
        assert self.keypoints is not None
        assert self.skeleton is not None

        annotations = [
            Annotation(keypoints=self.keypoints, skeleton=self.skeleton).set(kps, fixed_score=None)
            for kps in keypoint_sets
        ]

        self._confidences(field[0])
        self._regressions(field[1], field[2], field[3], field[4], confidence_fields=field[0], annotations=annotations)

    def predicted(self, field, *, annotations=None):
        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 5:7], field[:, 4], field[:, 8],
                          annotations=annotations,
                          confidence_fields=field[:, 0],
                          uv_is_offset=False)

    def _confidences(self, confidences):
        if not self.show_confidences:
            return

        indices = np.arange(confidences.shape[0])[np.nanmax(confidences, axis=(1,2))>0.3]
        for f in indices:
            LOG.debug('%s,%s',
                      self.keypoints[self.skeleton[f][0] - 1],
                      self.keypoints[self.skeleton[f][1] - 1])

            if self._meta:
                fig_file = os.path.join(self.fig_file, self._meta['file_name'].replace(".jpg", ".caf_c_"+str(f)+".jpg")).replace(".png", ".caf_c_"+str(f)+".png") if self.fig_file else None
            else:
                fig_file = os.path.join(self.fig_file, "prediction_image.caf_c_"+str(f)+".jpg").replace(".png", ".caf_c_"+str(f)+".png") if self.fig_file else None
            with self.image_canvas(self._processed_image, fig_file=fig_file) as ax:
                ax.text(0, 0, '{}'.format(self.keypoints[self.skeleton[f][1] - 1]), fontsize=8, color='red')
                im = ax.imshow(self.scale_scalar(confidences[f], self.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_BLUES_NAN)
                self.colorbar(ax, im)

    def _regressions(self, regression_fields1, regression_fields2,
                     scale_fields1, scale_fields2, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        if not self.show_regressions:
            return

        indices = np.arange(confidence_fields.shape[0])[np.nanmax(confidence_fields, axis=(1,2))>0.3]
        for f in indices:
            LOG.debug('%s,%s',
                      self.keypoints[self.skeleton[f][0] - 1],
                      self.keypoints[self.skeleton[f][1] - 1])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            if self._meta:
                fig_file = os.path.join(self.fig_file, self._meta['file_name'].replace(".jpg", ".caf_reg_"+str(f)+".jpg")).replace(".png", ".cif_reg_"+str(f)+".png") if self.fig_file else None
            else:
                fig_file = os.path.join(self.fig_file, "prediction_image.caf_reg_"+str(f)+".jpg").replace(".png", ".caf_reg_"+str(f)+".png") if self.fig_file else None
            with self.image_canvas(self._processed_image, fig_file=fig_file) as ax:
                show.white_screen(ax, alpha=0.5)
                if annotations:
                    self.keypoint_painter.annotations(ax, annotations)
                q1 = show.quiver(ax,
                                 regression_fields1[f, :2],
                                 confidence_field=confidence_field,
                                 xy_scale=self.stride, uv_is_offset=uv_is_offset,
                                 cmap='Blues', clim=(0.5, 1.0), width=0.001)
                show.quiver(ax,
                            regression_fields2[f, :2],
                            confidence_field=confidence_field,
                            xy_scale=self.stride, uv_is_offset=uv_is_offset,
                            cmap='Greens', clim=(0.5, 1.0), width=0.001)
                show.boxes(ax, scale_fields1[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields1[f, :2],
                           xy_scale=self.stride, cmap='Blues', fill=False,
                           regression_field_is_offset=uv_is_offset)
                show.boxes(ax, scale_fields2[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields2[f, :2],
                           xy_scale=self.stride, cmap='Greens', fill=False,
                           regression_field_is_offset=uv_is_offset)
                if self.show_margin:
                    show.margins(ax, regression_fields1[f, :6], xy_scale=self.stride)
                    show.margins(ax, regression_fields2[f, :6], xy_scale=self.stride)

                self.colorbar(ax, q1)
