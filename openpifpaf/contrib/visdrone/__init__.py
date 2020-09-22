import openpifpaf

from . import datamodule


def register():
    openpifpaf.DATAMODULES['visdrone'] = datamodule.VisDroneModule
