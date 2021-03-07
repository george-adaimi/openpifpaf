import openpifpaf

from . import basenet


def register():

    openpifpaf.BASE_TYPES.add(basenet.HRNet)
    openpifpaf.BASE_FACTORIES['hrnetw32v2'] = lambda: basenet.HRNet(cfg_file="w32_384x288_adam_lr1e-3.yaml", name='hrnetw32v2', detection=True)
    openpifpaf.BASE_FACTORIES['hrnetw48v2'] = lambda: basenet.HRNet(cfg_file="w48_384x288_adam_lr1e-3.yaml", name='hrnetw48v2', detection=True)
    openpifpaf.BASE_FACTORIES['hrnetw32v1'] = lambda: basenet.HRNet(cfg_file="w32_384x288_adam_lr1e-3.yaml", name='hrnetw32v1', detection=False)
    openpifpaf.BASE_FACTORIES['hrnetw48v1'] = lambda: basenet.HRNet(cfg_file="w48_384x288_adam_lr1e-3.yaml", name='hrnetw48v1', detection=False)  # pylint: disable=unnecessary-lambda
