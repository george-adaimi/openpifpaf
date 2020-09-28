import openpifpaf

from . import headmeta
from .visual_relationship import datamodule
from .cifdetraf import CifDetRaf

def register():
    openpifpaf.HEAD_FACTORIES[headmeta.Raf] = openpifpaf.network.heads.CompositeField3
    openpifpaf.DATAMODULES['visual_relationship'] = datamodule.VisualRelationshipModule
    openpifpaf.DECODERS.add(CifDetRaf)
