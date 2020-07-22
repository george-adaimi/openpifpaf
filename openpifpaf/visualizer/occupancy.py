import logging
import numpy as np
import os

from .base import BaseVisualizer

LOG = logging.getLogger(__name__)


class Occupancy(BaseVisualizer):
    show = False
    fig_file = None

    def __init__(self, *, field_names=None):
        super().__init__('occupancy')
        self.field_names = field_names

    def predicted(self, occupancy):
        if not self.show:
            return

        indices = np.arange(occupancy.occupancy.shape[0])[np.nanmax(occupancy.occupancy, axis=(1,2))>0.3]
        for f in indices:
            LOG.debug('%d (field name: %s)',
                      f, self.field_names[f] if self.field_names else 'unknown')

            # occupancy maps are at a reduced scale wrt the processed image
            reduced_image = self._processed_image[::occupancy.reduction, ::occupancy.reduction]

            if self._meta:
                fig_file = os.path.join(self.fig_file, self._meta['file_name'].replace(".jpg", ".occ_"+str(f)+".jpg")).replace(".png", ".occ_"+str(f)+".png") if self.fig_file else None
            else:
                fig_file = os.path.join(self.fig_file, "prediction_image.occ_"+str(f)+".jpg").replace(".png", ".occ_"+str(f)+".png") if self.fig_file else None
            with self.image_canvas(reduced_image, fig_file=fig_file) as ax:
                occ = occupancy.occupancy[f].copy()
                occ[occ > 0] = 1.0
                ax.imshow(occ, alpha=0.7)
