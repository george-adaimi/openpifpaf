import logging
import os
import numpy as np

from openpifpaf.visualizer import Base
from .annotation import AnnotationRaf
from openpifpaf import show
from . import headmeta
from .painters import RelationPainter

LOG = logging.getLogger(__name__)


class Raf(Base):
    show_margin = False
    show_background = False
    show_confidences = False
    show_regressions = False
    fig_file = None

    def __init__(self, meta: headmeta.Raf):
        super().__init__(meta.name)
        self.meta = meta
        self.detection_painter = RelationPainter(xy_scale=meta.stride)

    def targets(self, field, *, annotation_dicts):
        annotations = []
        dict_id2index = {}
        for det_index, ann in enumerate(annotation_dicts):
            dict_id2index[ann['detection_id']] = det_index

        for ann in annotation_dicts:
            if ann['iscrowd'] or not np.any(ann['bbox']) or not len(ann['object_index']) > 0:
                continue
            bbox = ann['bbox']
            category_id = ann['category_id']
            for object_id, predicate in zip(ann['object_index'], ann['predicate']):
                if not(object_id in dict_id2index) or annotation_dicts[dict_id2index[object_id]]['iscrowd']:
                    continue
                object_idx = dict_id2index[object_id]
                bbox_object = annotation_dicts[object_idx]['bbox']
                object_category = annotation_dicts[object_idx]['category_id']
                annotations.append(AnnotationRaf(self.meta.obj_categories,
                                    self.meta.rel_categories).set(
                                    object_category, category_id,
                                    predicate+1, None, None, None, bbox, bbox_object
                                    ))
        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 3:5], field[:, 5], field[:, 6], confidence_fields=field[:,0], annotations=annotations)

    def predicted(self, field):
        self._confidences(field[:, 0])
        if field.shape[1]>=9:

            self._regressions(field[:, 1:3], field[:, 3:5], field[:, 7], field[:, 8],
                              annotations=self._ground_truth,
                              confidence_fields=field[:, 0],
                              uv_is_offset=False)
        else:
            self._regressions(field[:, 1:3], field[:, 3:5], None, None,
                              annotations=self._ground_truth,
                              confidence_fields=field[:, 0],
                              uv_is_offset=False)

    def _confidences(self, confidences):
        if not self.show_confidences:
            return
        indices = np.arange(confidences.shape[0])[np.nanmax(confidences, axis=(1,2))>0.2]

        for f in indices:
            LOG.debug('%s', self.meta.rel_categories[f])

            with self.image_canvas(self._processed_image) as ax:
                ax.text(0, 0, '{}'.format(self.meta.rel_categories[f]), fontsize=8, color='red')
                im = ax.imshow(self.scale_scalar(confidences[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap='Blues')
                self.colorbar(ax, im)

    def _regressions(self, regression_fields1, regression_fields2,
                     scale_fields1, scale_fields2, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):
        if not self.show_regressions:
            return
        indices = np.arange(confidence_fields.shape[0])[np.nanmax(confidence_fields, axis=(1,2))>0.2]
        for f in indices:
            LOG.debug('%s', self.meta.rel_categories[f])

            confidence_field = confidence_fields[f] if confidence_fields is not None else None
            with self.image_canvas(self._processed_image) as ax:
                show.white_screen(ax, alpha=0.5)
                ax.text(0, 0, '{}'.format(self.meta.rel_categories[f]), fontsize=14, color='red')
                if annotations:
                    self.detection_painter.annotations(ax, annotations, color=('lightgray', 'mediumblue', 'firebrick'))
                q1 = show.quiver(ax,
                                 regression_fields1[f, :2],
                                 confidence_field=confidence_field,
                                 xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                                 cmap='Reds', clim=(0.5, 1.0), width=0.001, threshold=0.2)
                show.quiver(ax,
                            regression_fields2[f, :2],
                            confidence_field=confidence_field,
                            xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                            cmap='Greens', clim=(0.5, 1.0), width=0.001, threshold=0.2)
                if scale_fields1 is not None and scale_fields2 is not None:
                    show.boxes(ax, scale_fields1[f] / 2.0,
                               confidence_field=confidence_field,
                               regression_field=regression_fields1[f, :2],
                               xy_scale=self.meta.stride, cmap='Reds', fill=False,
                               regression_field_is_offset=uv_is_offset)
                    show.boxes(ax, scale_fields2[f] / 2.0,
                               confidence_field=confidence_field,
                               regression_field=regression_fields2[f, :2],
                               xy_scale=self.meta.stride, cmap='Greens', fill=False,
                               regression_field_is_offset=uv_is_offset, threshold=0.2)
                if self.show_margin:
                    show.margins(ax, regression_fields1[f, :6], xy_scale=self.meta.stride)
                    show.margins(ax, regression_fields2[f, :6], xy_scale=self.meta.stride)

                self.colorbar(ax, q1)
