import openpifpaf

from . import headmeta
from .visual_relationship import datamodule
from .cifdetraf import CifDetRaf
from .painters import RelationPainter
from .annotation import AnnotationRaf, annotation_inverse
from .coco_raf.datamodule import CocoDet
from .visual_genome.datamodule import VGModule
from .heads import DeepCompositeField3

def register():
    openpifpaf.HEAD_FACTORIES[headmeta.Raf] = DeepCompositeField3 #openpifpaf.network.heads.CompositeField3
    openpifpaf.HEAD_FACTORIES[headmeta.CifDet_deep] = DeepCompositeField3
    openpifpaf.DATAMODULES['visual_relationship'] = datamodule.VisualRelationshipModule
    openpifpaf.DECODERS.add(CifDetRaf)
    openpifpaf.PAINTERS['AnnotationRaf'] = RelationPainter
    openpifpaf.PREPROCESS_INVERSE[AnnotationRaf] = annotation_inverse

    openpifpaf.DATAMODULES['cocodet_raf'] = CocoDet
    openpifpaf.DATAMODULES['vg'] = VGModule
