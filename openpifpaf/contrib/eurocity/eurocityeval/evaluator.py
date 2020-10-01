import os
from collections import defaultdict

from . import statistics
from .match import Evaluator, Result, compare_all
from .params import ParamsFactory, IoU


def create_evaluator(data, difficulty, ignore_other_vru, type='pedestrian'):
    if type == 'pedestrian':
        params = ParamsFactory(difficulty=difficulty,
                               ignore_other_vru=ignore_other_vru,
                               tolerated_other_classes=['rider'],
                               dont_care_classes=['person-group-far-away'],
                               detections_type=['pedestrian'],
                               ignore_type_for_skipped_gts=1,
                               size_limits={'reasonable': 40, 'small': 30,
                                            'occluded': 40, 'all': 20},
                               occ_limits={'reasonable': 40, 'small': 40,
                                           'occluded': 80, 'all': 80},
                               size_upper_limits={'small': 60},
                               occ_lower_limits={'occluded': 40},
                               discard_depictions=True,
                               clipping_boxes=True,
                               transform_det_to_xy_coordinates=True
                               )
    elif type == 'rider':
        params = ParamsFactory(difficulty=difficulty,
                               ignore_other_vru=ignore_other_vru,
                               tolerated_other_classes=['pedestrian'],
                               dont_care_classes=['rider+vehicle-group-far-away'],
                               detections_type=['rider'],
                               ignore_type_for_skipped_gts=1,
                               size_limits={'reasonable': 40, 'small': 30,
                                            'occluded': 40, 'all': 20},
                               occ_limits={'reasonable': 40, 'small': 40,
                                           'occluded': 80, 'all': 80},
                               size_upper_limits={'small': 60},
                               occ_lower_limits={'occluded': 40},
                               discard_depictions=True,
                               clipping_boxes=True,
                               transform_det_to_xy_coordinates=True,
                               rider_boxes_including_vehicles=True
                               )
    else:
        assert False, 'Evaluation type not supported'

    return Evaluator(data,
                      metric=IoU,
                      comparable_identities=compare_all,
                      ignore_gt=params.ignore_gt,
                      skip_gt=params.skip_gt,
                      skip_det=params.skip_det,
                      preprocess_gt=params.preprocess_gt,
                      preprocess_det=params.preprocess_det,
                      allow_multiple_matches=False)


def evaluate(difficulty, ignore_other_vru, data, use_cache, eval_type='pedestrian'):

    """The actual evaluation"""
    evaluator = create_evaluator(data, difficulty, ignore_other_vru, eval_type)
    result = evaluator.result

    # Miss Rate vs False Positive Per Image
    mr_fppi = statistics.MrFppi(result=result)
    title = 'difficulty={}, ignore_other_vru={}, evaltype={}'.format(difficulty, ignore_other_vru,
                                                                     eval_type)
    label = 'lamr: {}'.format(mr_fppi.log_avg_mr_reference_implementation())
    fig = mr_fppi.create_plot(title, label)
    filename = 'lamr_ignore={}_difficulty={}_evaltype={}'.format(ignore_other_vru, difficulty,
                                                                 eval_type)


    #fig.savefig(os.path.join(results_path, '{}.pdf'.format(filename)))  # vector graphic
    #fig.savefig(os.path.join(results_path, '{}.png'.format(filename)))  # png

    # print '# ----------------------------------------------------------------- #'
    # print 'Finished evaluation of ' + det_method_name
    # print 'difficulty={}, ignore_other_vru={}, evaltype={}'.format(difficulty, ignore_other_vru,
    #                                                                eval_type)
    # print '---'
    # print 'Log-Avg Miss Rate (caltech reference implementation): ', \
    #     mr_fppi.log_avg_mr_reference_implementation()
    # print '-'
    # print 'Processed number of frames: ', result.nof_imgs
    # print 'Number of ignored ground truth annotations: ', \
    #     len(result.gts_including_ignored) - len(result.gts)
    # print 'Number of skipped ground truth annotations: ', result.skipped_gts['count']
    # print 'Classes of skipped ground truth annotations: ', list(result.skipped_gts['types'])
    # print 'Number of skipped detections: ', result.skipped_dets['count']
    # print 'Classes of skipped detections: ', list(result.skipped_dets['types'])
    # print '# ----------------------------------------------------------------- #'
    # print ''

    return {'mr_fppi': mr_fppi.log_avg_mr_reference_implementation(),
            'ignored_gt': len(result.gts_including_ignored) - len(result.gts),
            'skipped_gt': result.skipped_gts['count'],
            'skipped_dets': result.skipped_dets['count'],
            'proc_frames': result.nof_imgs,
            'classes_skipped_gt': list(result.skipped_gts['types']),
            'classes_skipped_dets': list(result.skipped_dets['types']),}


def evaluate_detection(data, eval_type='pedestrian'):
    nested_results = lambda: defaultdict(nested_results)
    results = nested_results()
    for difficulty in ['reasonable', 'small', 'occluded', 'all']:
        # False is the default case used by the benchmark server,
        # use [True, False] if you want to compare the enforce with the ignore setting
        for ignore_other_vru in [True, False]:
            results[difficulty][str(ignore_other_vru).lower()] = evaluate(difficulty, ignore_other_vru, data,
                     use_cache=False, eval_type=eval_type)

    return results
# def eval(time='day', mode='val', eval_type='pedestrian'):
#     assert time in ['day', 'night']
#     assert mode in ['val', 'test']
#
#     gt_path = './data/{}/labels/{}'.format(time, mode)
#     det_path = './data/mock_detections/{}/{}'.format(time, mode)
#     det_method_name = 'Faster R-CNN'
#
#     # folder where you find all the results (unless you change other paths...)
#     results_path = os.path.abspath('./results')
#     if not os.path.exists(results_path):
#         os.makedirs(results_path)
#
#     evaluate_detection(results_path, det_path, gt_path, det_method_name, eval_type)
#     print ''
#     print '# -----------------------------------------------------------------'
#     print 'Finished evaluation, results can be found here: {}'.format(results_path)
#     print '# -----------------------------------------------------------------'
#
#
#     import matplotlib.pyplot as plt
#     plt.show()  # comment this if you don't want plots to pop up

# if __name__ == "__main__":
#     eval(time='day', mode='val', eval_type='pedestrian')
