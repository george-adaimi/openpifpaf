import openpifpaf
import argparse
import torch
import logging

from torch import nn
LOG = logging.getLogger(__name__)
from .hourglass import Hourglass, Conv, Pool, Residual

# class HeatmapLoss(torch.nn.Module):
#     """
#     loss for detection heatmap
#     """
#     def __init__(self):
#         super(HeatmapLoss, self).__init__()
#
#     def forward(self, pred, gt):
#         l = ((pred - gt)**2)
#         l = l.mean(dim=3).mean(dim=2).mean(dim=1)
#         return l ## l of dim bsize

class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()
        self.confidence_loss = openpifpaf.network.losses.Bce(focal_gamma=2.0, detach_focal=True)
        self.background_weight = 1.0
    def forward(self, x_confidence, t_confidence):
        bce_masks = torch.isnan(t_confidence).bitwise_not_()
        if not torch.any(bce_masks):
            return None

        batch_size = x_confidence.shape[0]
        LOG.debug('batch size = %d', batch_size)

        LOG.debug('BCE_intermediate: x = %s, target = %s, mask = %s',
                  x_confidence.shape, t_confidence.shape, bce_masks.shape)
        bce_target = torch.masked_select(t_confidence, bce_masks)
        x_confidence = torch.masked_select(x_confidence, bce_masks)
        ce_loss = self.confidence_loss(x_confidence, bce_target)
        if self.background_weight != 1.0:
            bce_weight = torch.ones_like(bce_target, requires_grad=False)
            bce_weight[bce_target == 0] *= self.background_weight
            ce_loss = ce_loss * bce_weight

        ce_loss = ce_loss.sum() / batch_size

        return ce_loss
class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class HGNet(nn.Module):
    nstack = 4
    def __init__(self, inp_dim, oup_dim,nstack, bn=False, increase=0,):
        super(HGNet, self).__init__()
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(self.nstack)] )

        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(self.nstack)] )

        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )

        self.heatmapLoss = HeatmapLoss()

    def stride(self):
        return 4

    def forward(self, x):
        x = self.pre(x)
        combined_hm_preds = []
        combined_feats = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            combined_feats.append(feature)
            if i < self.nstack - 1:
                preds = self.outs[i](feature)
                combined_hm_preds.append(preds)
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.mean(torch.stack(combined_feats,1), 1), torch.stack(combined_hm_preds, 1)

class HGNet_base(openpifpaf.network.BaseNetwork):
    nstack = 4
    lambdas = None
    def __init__(self, name, inp_dim=512, oup_dim=170):
        model = HGNet(inp_dim, oup_dim, nstack= self.nstack)
        super(HGNet_base, self).__init__(name, stride=model.stride(), out_features=inp_dim)
        self.model = model

        if not self.lambdas:
            self.lambdas = [1.0 for _ in range(self.nstack-1)]
        assert len(self.lambdas) == (self.nstack - 1)


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Commond line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('BaseNet for HourGlass')
        group.add_argument('--hg-nstack',
                           default=cls.nstack, type=int,
                           help='number of hourglass stack')
        group.add_argument('--intermediate-lambdas', default=None, type=float, nargs='+',
                           help='prefactor for head losses')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.nstack = args.hg_nstack
        cls.lambdas = args.intermediate_lambdas

    def calc_inter_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack-1):
            combined_loss.append(self.lambdas[i]*self.model.heatmapLoss(combined_hm_preds[:,i], heatmaps))
        return combined_loss
    def forward(self, x):
        return self.model(x)
