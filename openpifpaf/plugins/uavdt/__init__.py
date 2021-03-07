import openpifpaf

from . import datamodule

def register():
    openpifpaf.DATAMODULES['uavdt'] = datamodule.UAVDTModule
