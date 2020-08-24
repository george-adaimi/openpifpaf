from ..network.heads import AssociationMeta, DetectionMeta, IntensityMeta, RelationMeta
from .constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    COCO_PERSON_SKELETON,
    COCO_PERSON_SIGMAS,
    COCO_UPRIGHT_POSE,
    DENSER_COCO_PERSON_CONNECTIONS,
    KINEMATIC_TREE_SKELETON,
    BBOX_UPRIGHT_POSE,
    BBOX_KEYPOINTS,
    BBOX_SIGMAS,
    DENSER_BBOX_CONNECTIONS,
    BBOX_SKELETON,
    COCO_BBOX_KEYPOINTS,
    COCO_BBOX_SIGMAS,
    COCO_BBOX_UPRIGHT_POSE,
    COCO_BBOX_SKELETON

)

from .factory import dataset_meta

def factory(head_names):
    if head_names is None:
        return None
    return [factory_single(hn) for hn in head_names]


def factory_single(head_name):
    if 'cifdet' in head_name:
        return DetectionMeta(head_name, dataset_meta['categories'])
    if 'raf' in head_name:
        return RelationMeta(head_name, dataset_meta['categories'], dataset_meta['rel_categories'])
    if 'cif160' in head_name:
        return IntensityMeta(head_name,
                             COCO_BBOX_KEYPOINTS,
                             COCO_BBOX_SIGMAS,
                             COCO_BBOX_UPRIGHT_POSE,
                             COCO_BBOX_SKELETON)
    if 'pif' in head_name or 'cif' in head_name:
        return IntensityMeta(head_name,
                             COCO_KEYPOINTS,
                             COCO_PERSON_SIGMAS,
                             COCO_UPRIGHT_POSE,
                             COCO_PERSON_SKELETON)
    if 'caf25' in head_name:
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               DENSER_COCO_PERSON_CONNECTIONS,
                               sparse_skeleton=COCO_PERSON_SKELETON,
                               only_in_field_of_view=True)
    if 'caf16' in head_name:
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               KINEMATIC_TREE_SKELETON)
    if head_name == 'caf':
        return AssociationMeta(head_name,
                               COCO_KEYPOINTS,
                               COCO_PERSON_SIGMAS,
                               COCO_UPRIGHT_POSE,
                               COCO_PERSON_SKELETON)
    if head_name == 'caf80':
        return AssociationMeta(head_name,
                               COCO_BBOX_KEYPOINTS,
                               COCO_BBOX_SIGMAS,
                               COCO_BBOX_UPRIGHT_POSE,
                               COCO_BBOX_SKELETON)
    raise NotImplementedError
