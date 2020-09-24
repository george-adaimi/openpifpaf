import openpifpaf

from . import generator


def register():
    openpifpaf.DECODERS.add(generator.ButterflyDet)
