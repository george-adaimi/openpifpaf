import openpifpaf
import argparse
from .hrnet import HighResolutionNet
class HRNet(openpifpaf.network.BaseNetwork):
    detection_channels = 256
    def __init__(self, cfg_file, name, detection=False, is_train=False):
        model = HighResolutionNet(cfg_file=cfg_file, detection=detection, detection_channels= self.detection_channels)
        model.init_weights(model.cfg['MODEL']['PRETRAINED'])
        super(HRNet, self).__init__(name, stride=model.stride(), out_features=model.stage4_cfg['NUM_CHANNELS'][0] if not detection else self.detection_channels)
        self.model = model

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Commond line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('BaseNet for HRNet')

        group.add_argument('--hrnet-detection-channels',
                           default=cls.detection_channels, type=int,
                           help='number of output channels for detection head')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.detection_channels = args.hrnet_detection_channels

    def forward(self, x):
        return self.model(x)
