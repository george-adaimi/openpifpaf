import logging
import os
import numpy as np

from .base import BaseVisualizer
from ..annotation import AnnotationDet
from .. import show

LOG = logging.getLogger(__name__)


class Raf(BaseVisualizer):
    show_margin = False
    show_background = False
    show_confidences = False
    show_regressions = False
    fig_file = None
    def __init__(self, head_name, *, stride=1, obj_categories=None, rel_categories=None):
        super().__init__(head_name)

        self.stride = stride
        self.obj_categories = obj_categories
        self.rel_categories = rel_categories
        self.detection_painter = show.DetectionPainter(xy_scale=stride)

    def targets(self, field, detections):
        #assert self.keypoints is not None
        #assert self.skeleton is not None

        annotations = [
            AnnotationDet(self.obj_categories).set(det['category_id'], None, det['bbox'])
            for k, det in detections.items()
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
            #LOG.debug('%s,%s',
            #          self.keypoints[self.skeleton[f][0] - 1],
            #          self.keypoints[self.skeleton[f][1] - 1])
            fig_file = os.path.join(self.fig_file, "prediction_image.raf_c_"+str(f)+".jpg") if self.fig_file else None
            with self.image_canvas(self._processed_image, fig_file=fig_file) as ax:
                ax.text(0, 0, '{}'.format(self.rel_categories[f]), fontsize=8, color='red')
                im = ax.imshow(self.scale_scalar(confidences[f], self.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap='Blues')
                self.colorbar(ax, im)

    def _regressions(self, regression_fields1, regression_fields2,
                     scale_fields1, scale_fields2, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        if not self.show_regressions:
            return
        indices = np.arange(confidence_fields.shape[0])[np.nanmax(confidence_fields, axis=(1,2))>0.3]
        for f in indices:
            #LOG.debug('%s,%s',
            #          self.keypoints[self.skeleton[f][0] - 1],
            #          self.keypoints[self.skeleton[f][1] - 1])

            confidence_field = confidence_fields[f] if confidence_fields is not None else None
            fig_file = os.path.join(self.fig_file, "prediction_image.raf_reg_"+str(f)+".jpg") if self.fig_file else None
            with self.image_canvas(self._processed_image, fig_file=fig_file) as ax:
                show.white_screen(ax, alpha=0.5)
                ax.text(0, 0, '{}'.format(self.rel_categories[f]), fontsize=14, color='red')
                if annotations:
                    self.detection_painter.annotations(ax, annotations)
                q1 = show.quiver(ax,
                                 regression_fields1[f, :2],
                                 confidence_field=confidence_field,
                                 xy_scale=self.stride, uv_is_offset=uv_is_offset,
                                 cmap='Reds', clim=(0.5, 1.0), width=0.001, threshold=0.2)
                show.quiver(ax,
                            regression_fields2[f, :2],
                            confidence_field=confidence_field,
                            xy_scale=self.stride, uv_is_offset=uv_is_offset,
                            cmap='Greens', clim=(0.5, 1.0), width=0.001, threshold=0.2)
                show.boxes(ax, scale_fields1[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields1[f, :2],
                           xy_scale=self.stride, cmap='Reds', fill=False,
                           regression_field_is_offset=uv_is_offset)
                show.boxes(ax, scale_fields2[f] / 2.0,
                           confidence_field=confidence_field,
                           regression_field=regression_fields2[f, :2],
                           xy_scale=self.stride, cmap='Greens', fill=False,
                           regression_field_is_offset=uv_is_offset, threshold=0.2)
                if self.show_margin:
                    show.margins(ax, regression_fields1[f, :6], xy_scale=self.stride)
                    show.margins(ax, regression_fields2[f, :6], xy_scale=self.stride)

                self.colorbar(ax, q1)
