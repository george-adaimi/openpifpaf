import openpifpaf

from . import basenet


def register():

    openpifpaf.BASE_TYPES.add(basenet.HGNet_base)
    openpifpaf.BASE_FACTORIES['hourglass'] = lambda: basenet.HGNet_base(name='hourglass', inp_dim=512, oup_dim=200)
