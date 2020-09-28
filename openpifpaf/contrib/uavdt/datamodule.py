import argparse
import numpy as np
import torch
import torchvision

import openpifpaf

from .uavdt import UAVDT
from .constants import BBOX_KEYPOINTS, BBOX_HFLIP
from . import metric

class UAVDTModule(openpifpaf.datasets.DataModule):
    train_image_dir = "data/UAV-benchmark-M/train/"
    val_image_dir = "data/UAV-benchmark-M/test/"
    eval_image_dir = val_image_dir
    train_annotations = "data/UAV-benchmark-M/GT/"
    val_annotations = "data/UAV-benchmark-M/GT/"
    eval_annotations = val_annotations

    debug = False
    pin_memory = False

    n_images = None
    square_edge = 512
    extended_scale = False
    orientation_invariant = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1

    eval_long_edge = None
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    def __init__(self):
        super().__init__()

        self.categories = ("vehicle")
        cifdet = openpifpaf.headmeta.CifDet('cifdet', 'uavdt', self.categories)
        cifdet.upsample_stride = self.upsample_stride
        self.head_metas = [cifdet,]


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module UAVDT')

        group.add_argument('--uavdt-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--uavdt-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--uavdt-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--uavdt-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--uavdt-n-images',
                           default=cls.n_images, type=int,
                           help='number of images to sample')
        group.add_argument('--uavdt-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--uavdt-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--uavdt-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        assert cls.augmentation
        group.add_argument('--uavdt-no-augmentation',
                           dest='uavdt_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--uavdt-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')

        group.add_argument('--uavdt-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # uavdt specific
        cls.train_annotations = args.uavdt_train_annotations
        cls.val_annotations = args.uavdt_val_annotations
        cls.train_image_dir = args.uavdt_train_image_dir
        cls.val_image_dir = args.uavdt_val_image_dir

        cls.n_images = args.uavdt_n_images
        cls.square_edge = args.uavdt_square_edge
        cls.extended_scale = args.uavdt_extended_scale
        cls.orientation_invariant = args.uavdt_orientation_invariant
        cls.augmentation = args.uavdt_augmentation
        cls.rescale_images = args.uavdt_rescale_images
        cls.upsample_stride = args.uavdt_upsample

    @staticmethod
    def _convert_data(parent_data, meta):
        image, category_id = parent_data

        anns = [{
            'bbox': np.asarray([5, 5, 21, 21], dtype=np.float32),
            'category_id': category_id + 1,
        }]

        return image, anns, meta

    def _preprocess(self):
        enc = openpifpaf.encoder.CifDet(self.head_metas[0])

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders([enc]),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.5 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.7 * self.rescale_images,
                             1.5 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        orientation_t = None
        if self.orientation_invariant:
            orientation_t = openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.RotateBy90(), self.orientation_invariant)

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.AnnotationJitter(),
            openpifpaf.transforms.RandomApply(openpifpaf.transforms.HFlip(BBOX_KEYPOINTS, BBOX_HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            orientation_t,
            openpifpaf.transforms.MinSize(min_side=4.0),
            openpifpaf.transforms.UnclippedArea(threshold=0.75),
            # transforms.UnclippedSides(),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders([enc]),
        ])

    def train_loader(self):
        train_data = UAVDT(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            n_images=self.n_images,
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = UAVDT(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            n_images=self.n_images,
        )

        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def _eval_preprocess(self):
        rescale_t = None
        if self.eval_extended_scale:
            assert self.eval_long_edge
            rescale_t = openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(self.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((self.eval_long_edge) // 2),
                ], salt=1)

        elif self.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(self.eval_long_edge)

        if self.batch_size == 1:
            #padding_t = [openpifpaf.transforms.CenterPadTight(16)]
            padding_t = openpifpaf.transforms.CenterPadTight(32)
        else:
            assert self.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(self.eval_long_edge)

        orientation_t = None
        if self.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                    None,
                    openpifpaf.transforms.RotateBy90(fixed_angle=90),
                    openpifpaf.transforms.RotateBy90(fixed_angle=180),
                    openpifpaf.transforms.RotateBy90(fixed_angle=270),
                ], salt=3)

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToDetAnnotations(self.categories),
                openpifpaf.transforms.ToCrowdAnnotations(self.categories),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = UAVDT(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            n_images=self.n_images,
            category_ids=[],
        )

        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [metric.UAVDT(
            self.eval_annotations,
            self.eval_image_dir,
        )]
