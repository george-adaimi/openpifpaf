import logging
import os
import numpy as np
import zipfile
import glob
from tqdm import tqdm
from PIL import Image

from openpifpaf.metric.base import Base
from .eurocityeval import evaluate_detection
LOG = logging.getLogger(__name__)

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class EuroCity(Base):
    def load_data_ecp(self, gt_path, gt_ext='.json'):
        gt_all = {}
        def _prepare_ecp_gt(gt):
            def translate_ecp_pose_to_image_coordinates(angle):
                angle = angle + 90.0

                # map to interval [0, 360)
                angle = angle % 360

                if angle > 180:
                    # map to interval (-180, 180]
                    angle -= 360.0

                return np.deg2rad(angle)

            orient = None
            if gt['identity'] == 'rider':
                if len(gt['children']) > 0:  # vehicle is annotated
                    for cgt in gt['children']:
                        if cgt['identity'] in ['bicycle', 'buggy', 'motorbike', 'scooter', 'tricycle',
                                               'wheelchair']:
                            orient = cgt.get('Orient', None) or cgt.get('orient', None)
            else:
                orient = gt.get('Orient', None) or gt.get('orient', None)

            if orient:
                gt['orient'] = translate_ecp_pose_to_image_coordinates(orient)
                gt.pop('Orient', None)

        def get_gt_frame(gt_dict):
            if gt_dict['identity'] == 'frame':
                pass
            elif '@converter' in gt_dict:
                gt_dict = gt_dict['children'][0]['children'][0]
            elif gt_dict['identity'] == 'seqlist':
                gt_dict = gt_dict['children']['children']

            # check if json layout is corrupt
            assert gt_dict['identity'] == 'frame'
            return gt_dict

        if not os.path.isdir(gt_path) and not gt_path.endswith('.dataset'):
            raise IOError('{} is not a directory and not a dataset file.'.format(gt_path))

        if gt_path.endswith('.dataset'):
            with open(gt_path) as datasetf:
                gt_files = datasetf.readlines()
                gt_files = [f.strip() for f in gt_files if len(f) > 0]
        else:
            gt_files = glob.glob(gt_path + '/*' + gt_ext)
            if not gt_files:
                gt_files = glob.glob(gt_path + '/*/*' + gt_ext)

        gt_files.sort()

        if not gt_files:
            raise ValueError('ERROR: No ground truth files found at given location! ABORT.'
                             'Given path was: {} and gt ext looked for was {}'.format(gt_path, gt_ext))

        for gt_file in gt_files:
            gt_fn = os.path.basename(gt_file)

            gt_frame_id = re.search('(.*?)' + gt_ext, gt_fn).group(1)

            with open(gt_file, 'rb') as f:
                gt = json.load(f)
            gt_frame = get_gt_frame(gt)
            for gt in gt_frame['children']:
                _prepare_ecp_gt(gt)

            gt_all[gt_fn] = gt.copy()
        return gt_all

    def __init__(self, gt_dir, img_dir, categories):

        self.imgPath = img_dir
        self.predictions = {}
        self.image_ids = []
        self.gt = self.load_data_ecp(gt_dir)
        self.categories = categories


    def accumulate(self, predictions, image_meta, ground_truth=None):
        image_id = int(image_meta['image_id'])
        width, height = image_meta['width_height']
        self.image_ids.append(image_id)

        image_annotations = []
        for pred in predictions:
            pred_data = pred.json_data()
            categ = pred_data['category_id']
            x,y,w,h = pred_data['bbox']
            x1, x2 = np.clip([x, x+w], a_min=0, a_max=width)
            y1, y2 = np.clip([y, y+h], a_min=0, a_max=height)
            s = pred_data['score']
            image_annotations.append({'x0': x1,
                        'x1': x2,
                        'y0': y1,
                        'y1': y2,
                        'score': 2,
                        'identity': self.categories[categ-1],
                        'orient': 0.0})

        frame = {'identity': 'frame'}
        frame['children'] = image_annotations
        self.predictions[image_meta['file_name'][:-4]+'.json'] = frame


    def write_predictions(self, filename):
        mkdir_if_missing(filename)
        for imageName, frame in self.predictions.items():
            json.dump(frame, open(os.path.join(filename,imageName), 'w'), indent=1)

        LOG.info('wrote predictions to %s', filename)
        with zipfile.ZipFile(os.path.join(filename,'predictions')+ '.zip', 'w') as myzip:
            for imageName in self.predictions.keys():
                myzip.write(os.path.join(filename,imageName))
        LOG.info('wrote %s.zip', os.path.join(filename,'predictions')+ '.zip')

    def stats(self):
        data = []
        for key, item in self.predictions.items():
            data.append({'gt': self.gt[key], 'det': item})
        results_pedestrian = evaluate_detection(data, eval_type='pedestrian')
        results_rider = evaluate_detection(data, eval_type='rider')

        stats = []
        text_labels = []
        for difficulty in ['reasonable', 'small', 'occluded', 'all']:
            for ignore_other_vru in [True, False]:
                stats.append(results_pedestrian[difficulty][str(ignore_other_vru).lowercase()]['mr_fppi'])
                stats.append(results_rider[difficulty][str(ignore_other_vru).lowercase()]['mr_fppi'])
                text_labels.append('mr_fppi_{}_{}_pedestrian'.format(difficulty, 'ignore' if ignore_other_vru else 'noignore'))
                text_labels.append('mr_fppi_{}_{}_rider'.format(difficulty, 'ignore' if ignore_other_vru else 'noignore'))
        data = {
            'stats': stats,
            'text_labels': text_labels,
        }

        return data
