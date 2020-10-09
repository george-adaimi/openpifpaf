import openpifpaf

from . import datamodule


def register():
    openpifpaf.DATAMODULES['cocodetkp'] = datamodule.CocoDetKp
