import argparse
import numpy as np
import torch
import torchvision

import openpifpaf

from .eurocity import EuroCity
from .constants import BBOX_KEYPOINTS, BBOX_HFLIP
from . import metric
from ..butterflydet.fullbutterfly import FullButterfly

class EuroCityModule(openpifpaf.datasets.DataModule):
    train_image_dir = "./data/ECP/{}/img/train"
    val_image_dir = "./data/ECP/{}/img/val"
    eval_image_dir = val_image_dir
    train_annotations = "./data/ECP/{}/labels/train"
    val_annotations = "./data/ECP/{}/labels/val"
    eval_annotations = val_annotations

    debug = False
    pin_memory = False

    n_images = None
    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1

    eval_long_edge = None
    eval_orientation_invariant = 0.0
    eval_extended_scale = False
    categories = ['pedestrian', 'rider']
    extra_categories = ['scooter', 'motorbike', 'bicycle', 'buggy', 'wheelchair', 'tricycle']

    full_butterfly = False
    rider_vehicles = False
    time = ['day', 'night']
    def __init__(self):
        super().__init__()

        if self.rider_vehicles:
            self.categories += extra_categories
        cifdet = openpifpaf.headmeta.CifDet('cifdet', 'eurocity', self.categories)
        cifdet.upsample_stride = self.upsample_stride
        self.head_metas = [cifdet,]


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module EuroCity')

        group.add_argument('--eurocity-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--eurocity-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--eurocity-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--eurocity-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--eurocity-n-images',
                           default=cls.n_images, type=int,
                           help='number of images to sample')
        group.add_argument('--eurocity-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--eurocity-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--eurocity-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        parser.add_argument('--eurocity-split', choices=('val', 'test'), default='val',
                            help='dataset to evaluate')
        assert cls.augmentation
        group.add_argument('--eurocity-no-augmentation',
                           dest='eurocity_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--eurocity-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')

        group.add_argument('--eurocity-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--eurocity-full-butterfly',
                           default=False, action='store_true',
                           help='use full butterfly')
        group.add_argument('--eurocity-rider-vehicles',
                           default=False, action='store_true',
                           help='detect vehicles of riders')
        group.add_argument('--eurocity-time', type=str, action='store',
                           choices=['day', 'night', 'both'],
                           default='both',
                           help='time to use (day or night)')
    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # eurocity specific
        cls.train_annotations = args.eurocity_train_annotations
        cls.val_annotations = args.eurocity_val_annotations
        cls.train_image_dir = args.eurocity_train_image_dir
        cls.val_image_dir = args.eurocity_val_image_dir

        if args.eurocity_split == 'test':
            cls.eval_image_dir = "./data/ECP/{}/img/test"
            cls.eval_annotations = None
        cls.n_images = args.eurocity_n_images
        cls.square_edge = args.eurocity_square_edge
        cls.extended_scale = args.eurocity_extended_scale
        cls.orientation_invariant = args.eurocity_orientation_invariant
        cls.augmentation = args.eurocity_augmentation
        cls.rescale_images = args.eurocity_rescale_images
        cls.upsample_stride = args.eurocity_upsample
        cls.full_butterfly = args.eurocity_full_butterfly
        cls.rider_vehicles = args.eurocity_rider_vehicles
        cls.time = [args.eurocity_time]
        if args.eurocity_time == 'both':
            cls.time = ['day', 'night']

    @staticmethod
    def _convert_data(parent_data, meta):
        image, category_id = parent_data

        anns = [{
            'bbox': np.asarray([5, 5, 21, 21], dtype=np.float32),
            'category_id': category_id + 1,
        }]

        return image, anns, meta

    def _preprocess(self):
        if self.full_butterfly:
            enc = FullButterfly(self.head_metas[0])
        else:
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
        train_data = EuroCity(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            time = self.time,
            preprocess=self._preprocess(),
            n_images=self.n_images,
            rider_vehicles=self.rider_vehicles,
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = EuroCity(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            time = self.time,
            preprocess=self._preprocess(),
            n_images=self.n_images,
            rider_vehicles=self.rider_vehicles,
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
        padding_t = None
        if self.batch_size == 1:
            #padding_t = openpifpaf.transforms.CenterPadTight(16)
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
        eval_data = EuroCity(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            time = self.time,
            preprocess=self._eval_preprocess(),
            n_images=self.n_images,
            rider_vehicles=self.rider_vehicles,
            category_ids=[],
        )

        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [metric.EuroCity(
            self.eval_annotations,
            self.eval_image_dir,
            self.time
        )]
