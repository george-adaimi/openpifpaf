import openpifpaf

from . import datamodule


def register():
    openpifpaf.DATAMODULES['vr_fit'] = datamodule.VisualRelationshipModule
