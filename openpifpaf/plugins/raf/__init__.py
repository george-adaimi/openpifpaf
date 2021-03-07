import openpifpaf

from . import headmeta
from .visual_relationship import datamodule
from .visual_relationship.datamodule_det import VisualRelationshipDetModule
from .cifdetraf import CifDetRaf
from .painters import RelationPainter
from .annotation import AnnotationRaf
from .coco_raf.datamodule import CocoDet
from .visual_genome.datamodule import VGModule
from .vg_h5.datamodule import VGModule as VGModule_h5
from .heads import DeepCompositeField3, DeepSharedCompositeField3
from .refinement_heads import RefinedCompositeField3
from .cocodet_deep import CocoDet as CocoDet_deep
from .losses import DCNCompositeLoss

def register():
    openpifpaf.HEADS[headmeta.Raf] = DeepCompositeField3 #openpifpaf.network.heads.CompositeField3
    openpifpaf.HEADS[headmeta.Raf_dcn] = RefinedCompositeField3#DeepCompositeField3 #openpifpaf.network.heads.CompositeField3
    openpifpaf.HEADS[headmeta.CifDet_deep] = DeepCompositeField3
    openpifpaf.HEADS[headmeta.CifDet_deepShared] = DeepSharedCompositeField3
    openpifpaf.DATAMODULES['visual_relationship'] = datamodule.VisualRelationshipModule
    openpifpaf.DATAMODULES['visual_relationship_det'] = VisualRelationshipDetModule
    openpifpaf.DECODERS.add(CifDetRaf)
    openpifpaf.PAINTERS['AnnotationRaf'] = RelationPainter

    openpifpaf.DATAMODULES['cocodet_raf'] = CocoDet
    openpifpaf.DATAMODULES['cocodet_deep'] = CocoDet_deep
    #openpifpaf.DATAMODULES['vg'] = VGModule
    openpifpaf.DATAMODULES['vg'] = VGModule_h5
    openpifpaf.LOSSES[headmeta.CifDet_deep] = openpifpaf.network.losses.CompositeLoss
    openpifpaf.LOSSES[headmeta.Raf] = openpifpaf.network.losses.CompositeLoss
    openpifpaf.LOSSES[headmeta.Raf_dcn] = DCNCompositeLoss
    #openpifpaf.LOSSES[headmeta.Raf_dcn] = openpifpaf.network.losses.CompositeLoss
    openpifpaf.LOSSES[headmeta.CifDet_deepShared] = openpifpaf.network.losses.CompositeLoss
