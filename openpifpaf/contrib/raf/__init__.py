import openpifpaf

from . import headmeta
from .visual_relationship import datamodule
from .cifdetraf import CifDetRaf
from .painters import RelationPainter
from .annotation import AnnotationRaf, annotation_inverse
def register():
    openpifpaf.HEAD_FACTORIES[headmeta.Raf] = openpifpaf.network.heads.CompositeField3
    openpifpaf.DATAMODULES['visual_relationship'] = datamodule.VisualRelationshipModule
    openpifpaf.DECODERS.add(CifDetRaf)
    openpifpaf.PAINTERS['AnnotationRaf'] = RelationPainter
    openpifpaf.PREPROCESS_INVERSE[AnnotationRaf] = annotation_inverse
