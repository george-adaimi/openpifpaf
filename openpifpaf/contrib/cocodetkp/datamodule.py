import argparse

import torch

from openpifpaf.datasets.module import DataModule
from openpifpaf import encoder, headmeta, metric, transforms
from openpifpaf.datasets.coco import Coco
from openpifpaf.datasets.collate import collate_images_anns_meta, collate_images_targets_meta
from openpifpaf.datasets.constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_PERSON_SIGMAS,
    COCO_UPRIGHT_POSE,
    DENSER_COCO_PERSON_CONNECTIONS,
    HFLIP,
)

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass


class CocoDetKp(DataModule):
    _test2017_annotations = 'data-mscoco/annotations/image_info_test2017.json'
    _testdev2017_annotations = 'data-mscoco/annotations/image_info_test-dev2017.json'
    _test2017_image_dir = 'data-mscoco/images/test2017/'

    # cli configurable
    train_annotations = 'data-mscoco/annotations/person_keypoints_train2017.json'
    val_annotations = 'data-mscoco/annotations/person_keypoints_val2017.json'
    eval_annotations = val_annotations
    train_image_dir = 'data-mscoco/images/train2017/'
    val_image_dir = 'data-mscoco/images/val2017/'
    eval_image_dir = val_image_dir

    square_edge = 385
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1

    eval_annotation_filter = True
    eval_long_edge = 641
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    def __init__(self):
        super().__init__()

        cif = headmeta.Cif('cif', 'cocodetkp',
                           keypoints=COCO_KEYPOINTS,
                           sigmas=COCO_PERSON_SIGMAS,
                           pose=COCO_UPRIGHT_POSE,
                           draw_skeleton=COCO_PERSON_SKELETON)
        caf = headmeta.Caf('caf', 'cocodetkp',
                           keypoints=COCO_KEYPOINTS,
                           sigmas=COCO_PERSON_SIGMAS,
                           pose=COCO_UPRIGHT_POSE,
                           skeleton=COCO_PERSON_SKELETON)
        dcaf = headmeta.Caf('caf25', 'cocodetkp',
                            keypoints=COCO_KEYPOINTS,
                            sigmas=COCO_PERSON_SIGMAS,
                            pose=COCO_UPRIGHT_POSE,
                            skeleton=DENSER_COCO_PERSON_CONNECTIONS,
                            sparse_skeleton=COCO_PERSON_SKELETON,
                            only_in_field_of_view=True)
        cifdet = headmeta.CifDet('cifdet', 'cocodetkp', COCO_CATEGORIES)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        dcaf.upsample_stride = self.upsample_stride
        cifdet.upsample_stride = self.upsample_stride

        self.head_metas = [cifdet, cif, caf, dcaf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module CocoDetKp')

        group.add_argument('--cocodetkp-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--cocodetkp-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--cocodetkp-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--cocodetkp-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--cocodetkp-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--cocodetkp-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--cocodetkp-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--cocodetkp-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--cocodetkp-no-augmentation',
                           dest='cocodetkp_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--cocodetkp-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--cocodetkp-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--cocodetkp-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')

        # evaluation
        eval_set_group = group.add_mutually_exclusive_group()
        eval_set_group.add_argument('--cocodetkp-eval-test2017', default=False, action='store_true')
        eval_set_group.add_argument('--cocodetkp-eval-testdev2017', default=False, action='store_true')

        assert cls.eval_annotation_filter
        assert not cls.eval_extended_scale

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocodetkp specific
        cls.train_annotations = args.cocodetkp_train_annotations
        cls.val_annotations = args.cocodetkp_val_annotations
        cls.train_image_dir = args.cocodetkp_train_image_dir
        cls.val_image_dir = args.cocodetkp_val_image_dir

        cls.square_edge = args.cocodetkp_square_edge
        cls.extended_scale = args.cocodetkp_extended_scale
        cls.orientation_invariant = args.cocodetkp_orientation_invariant
        cls.blur = args.cocodetkp_blur
        cls.augmentation = args.cocodetkp_augmentation
        cls.rescale_images = args.cocodetkp_rescale_images
        cls.upsample_stride = args.cocodetkp_upsample
        cls.min_kp_anns = args.cocodetkp_min_kp_anns

        # evaluation
        cls.eval_annotation_filter = args.coco_eval_annotation_filter
        if args.cocodetkp_eval_test2017:
            cls.eval_image_dir = cls._test2017_image_dir
            cls.eval_annotations = cls._test2017_annotations
            cls.annotation_filter = False
        if args.cocodetkp_eval_testdev2017:
            cls.eval_image_dir = cls._test2017_image_dir
            cls.eval_annotations = cls._testdev2017_annotations
            cls.annotation_filter = False
        cls.eval_long_edge = args.coco_eval_long_edge
        cls.eval_orientation_invariant = args.coco_eval_orientation_invariant
        cls.eval_extended_scale = args.coco_eval_extended_scale

        if (args.cocodetkp_eval_test2017 or args.cocodetkp_eval_testdev2017) \
            and not args.write_predictions and not args.debug:
            raise Exception('have to use --write-predictions for this dataset')

    def _preprocess(self):
        encoders = (encoder.CifDet(self.head_metas[0]),
                    encoder.Cif(self.head_metas[1]),
                    encoder.Caf(self.head_metas[2]),
                    encoder.Caf(self.head_metas[3]))

        if not self.augmentation:
            return transforms.Compose([
                transforms.NormalizeAnnotations(),
                transforms.RescaleAbsolute(self.square_edge),
                transforms.CenterPad(self.square_edge),
                transforms.EVAL_TRANSFORM,
                transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.4 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        blur_t = None
        if self.blur:
            blur_t = transforms.RandomApply(transforms.Blur(), self.blur)

        orientation_t = None
        if self.orientation_invariant:
            orientation_t = transforms.RandomApply(
                transforms.RotateBy90(), self.orientation_invariant)

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.AnnotationJitter(),
            transforms.RandomApply(transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            blur_t,
            transforms.Crop(self.square_edge, use_area_of_interest=True),
            transforms.CenterPad(self.square_edge),
            orientation_t,
            transforms.TRAIN_TRANSFORM,
            transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = Coco(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    def val_loader(self):
        val_data = Coco(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                transforms.DeterministicEqualChoice([
                    transforms.RescaleAbsolute(cls.eval_long_edge),
                    transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = transforms.DeterministicEqualChoice([
                None,
                transforms.RotateBy90(fixed_angle=90),
                transforms.RotateBy90(fixed_angle=180),
                transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return transforms.Compose([
            *self.common_eval_preprocess(),
            transforms.ToAnnotations([
                transforms.ToKpAnnotations(
                    COCO_CATEGORIES,
                    keypoints_by_category={1: self.head_metas[1].keypoints},
                    skeleton_by_category={1: self.head_metas[2].skeleton},
                ),
                transforms.ToCrowdAnnotations(COCO_CATEGORIES),
            ]),
            transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = Coco(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[1] if self.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=collate_images_anns_meta)

    def metrics(self):
        return [metric.Coco(
            pycocotools.coco.COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[1],
            iou_type='keypoints',
        ),
        metric.Coco(
        pycocotools.coco.COCO(self.eval_annotations),
        max_per_image=100,
        category_ids=[],
        iou_type='bbox')]
