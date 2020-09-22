import openpifpaf
from .hrnet import HighResolutionNet
class HRNet(openpifpaf.network.BaseNetwork):
    def __init__(self, cfg_file, name, detection=False, is_train=False):
        model = HighResolutionNet(cfg_file=cfg_file, detection=detection)
        if is_train:
            model.init_weights(model.cfg['MODEL']['PRETRAINED'])
        super(HRNet, self).__init__(name, stride=model.stride(), out_features=model.stage4_cfg['NUM_CHANNELS'][0] if not detection else 512)
        self.model = model

    def forward(self, x):
        return self.model(x)
