import openpifpaf

from . import datamodule


def register():
    openpifpaf.DATAMODULES['eurocity'] = datamodule.EuroCityModule
