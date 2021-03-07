import torch
import argparse
import glob
import json
import logging
import os

import PIL
import torch

from openpifpaf import eval, datasets, decoder, network

def apply(f, items):
    """Apply f in a nested fashion to all items that are not list or tuple."""
    if items is None:
        return None
    if isinstance(items, (list, tuple)):
        return [apply(f, i) for i in items]
    return f(items)

def main():
    args = eval.cli()

    datamodule = datasets.factory(args.dataset)
    net_cpu, start_epoch = network.factory_from_args(args, head_metas=datamodule.head_metas)
    head_metas = [hn.meta for hn in net_cpu.head_nets]
    processor = decoder.factory(
        head_metas, profile=args.profile_decoder, profile_device=args.device)


    train_loader = datamodule.debugger_loader()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        with torch.autograd.profiler.record_function('tonumpy'):
            heads = apply(lambda x: x.cpu().numpy(), target)

        # index by frame (item in batch)
        head_iter = apply(iter, heads)
        heads = []
        while True:
            try:
                heads.append(apply(next, head_iter))
            except StopIteration:
                break
        pred_batch = processor(heads[0])
        import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
