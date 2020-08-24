import logging
import numpy as np
import os

from .base import BaseVisualizer
from ..annotation import Annotation
from .. import show

LOG = logging.getLogger(__name__)


class Cif(BaseVisualizer):
    show_margin = False
    show_confidences = False
    show_regressions = False
    show_background = False
    fig_file = None

    def __init__(self, head_name, *, stride=1, keypoints=None, skeleton=None):
        super().__init__(head_name)

        self.stride = stride
        self.keypoints = keypoints
        self.skeleton = skeleton

        self.keypoint_painter = show.KeypointPainter()  #xy_scale=self.stride)

    def targets(self, field, *, annotation_dicts):
        assert self.keypoints is not None
        assert self.skeleton is not None

        annotations = [
            Annotation(keypoints=self.keypoints, skeleton=self.skeleton).set(
                ann['keypoints'], fixed_score=None, fixed_bbox=ann['bbox'])
            for ann in annotation_dicts
        ]

        self._confidences(field[0])
        self._regressions(field[1], field[2], confidence_fields=field[0], annotations=annotations)

    def predicted(self, field, *, annotations=None):
        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 4],
                          annotations=annotations,
                          confidence_fields=field[:, 0],
                          uv_is_offset=False)

    def _confidences(self, confidences):
        if not self.show_confidences:
            return
        indices = np.arange(confidences.shape[0])[np.nanmax(confidences, axis=(1,2))>0.3]
        for f in indices:
            LOG.debug('%s', self.keypoints[f])

            if self._meta:
                fig_file = os.path.join(self.fig_file, self._meta['file_name'].replace(".jpg", ".cif_c_"+str(f)+".jpg")).replace(".png", ".cif_c_"+str(f)+".png") if self.fig_file else None
            else:
                fig_file = os.path.join(self.fig_file, "prediction_image.cif_c_"+str(f)+".jpg").replace(".png", ".cif_c_"+str(f)+".png") if self.fig_file else None
            with self.image_canvas(self._processed_image, fig_file=fig_file) as ax:
                ax.text(0, 0, '{}'.format(self.keypoints[f]), fontsize=8, color='red')
                im = ax.imshow(self.scale_scalar(confidences[f], self.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap='Oranges')
                self.colorbar(ax, im)

    def _regressions(self, regression_fields, scale_fields, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        if not self.show_regressions:
            return
        indices = np.arange(confidence_fields.shape[0])[np.nanmax(confidence_fields, axis=(1,2))>0.3]
        for f in indices:
            LOG.debug('%s', self.keypoints[f])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            if self._meta:
                fig_file = os.path.join(self.fig_file, self._meta['file_name'].replace(".jpg", ".cif_reg_"+str(f)+".jpg")).replace(".png", ".cif_reg_"+str(f)+".png") if self.fig_file else None
            else:
                fig_file = os.path.join(self.fig_file, "prediction_image.cif_reg_"+str(f)+".jpg").replace(".png", ".cif_reg_"+str(f)+".png") if self.fig_file else None
            with self.image_canvas(self._processed_image, fig_file=fig_file) as ax:
                show.white_screen(ax, alpha=0.5)
                if annotations:
                    self.keypoint_painter.annotations(ax, annotations)
                q = show.quiver(ax,
                                regression_fields[f, :2],
                                confidence_field=confidence_field,
                                xy_scale=self.stride, uv_is_offset=uv_is_offset,
                                cmap='Oranges', clim=(0.5, 1.0), width=0.001)
                show.boxes(ax, scale_fields[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields[f, :2],
                           xy_scale=self.stride, cmap='Oranges', fill=False,
                           regression_field_is_offset=uv_is_offset)
                if self.show_margin:
                    show.margins(ax, regression_fields[f, :6], xy_scale=self.stride)

                self.colorbar(ax, q)
