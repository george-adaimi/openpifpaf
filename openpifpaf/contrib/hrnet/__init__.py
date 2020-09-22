import openpifpaf

from . import basenet


def register():

    openpifpaf.BASE_TYPES.add(basenet.HRNet)
    openpifpaf.BASE_FACTORIES['hrnetw32det'] = lambda: basenet.HRNet(cfg_file="w32_384x288_adam_lr1e-3.yaml", name='hrnetw32det', detection=True)
    openpifpaf.BASE_FACTORIES['hrnetw48det'] = lambda: basenet.HRNet(cfg_file="w48_384x288_adam_lr1e-3.yaml", name='hrnetw48det', detection=True)
    openpifpaf.BASE_FACTORIES['hrnetw32'] = lambda: basenet.HRNet(cfg_file="w32_384x288_adam_lr1e-3.yaml", name='hrnetw32', detection=False)
    openpifpaf.BASE_FACTORIES['hrnetw48'] = lambda: basenet.HRNet(cfg_file="w48_384x288_adam_lr1e-3.yaml", name='hrnetw48', detection=False)  # pylint: disable=unnecessary-lambda
