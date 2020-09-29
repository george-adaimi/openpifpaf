import logging
import os
import numpy as np
import zipfile
from tqdm import tqdm
from PIL import Image

from openpifpaf.metric.base import Base
from uavdteval import CalculateDetectionPR_overall2, CalculateDetectionPR_seq2

LOG = logging.getLogger(__name__)

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class UAVDT(Base):
    #text_labels = ['AP', 'AP_vehicle-categ', 'AP_vehicle-occl', 'AP_out-of-view']
    text_labels = []
    def readFile(self, filepath):
        result = []
        with open(filepath) as f:
            for line in f:
                result.append([float(elem) for elem in line.split(',')])

        return np.asarray(result)

    def __init__(self, gt_dir, img_dir):

        self.imgPath = img_dir
        self.predictions = {}
        self.image_ids = []
        self.gt = {}
        self.gt_ign = {}
        self.gt_dir = gt_dir
        for fileName in os.listdir(gt_dir):
            if '_gt_whole.txt' in fileName:
                self.gt[fileName[:-13]] = self.readFile(os.path.join(gt_dir, fileName))
            elif '_gt_ignore.txt' in fileName:
                self.gt_ign[fileName[:-14]] = self.readFile(os.path.join(gt_dir, fileName))


    def accumulate(self, predictions, image_meta, ground_truth=None):
        image_id = int(image_meta['image_id'])
        width, height = image_meta['width_height']
        self.image_ids.append(image_id)

        image_annotations = []
        fileName = image_meta['file_dir']
        fileName = fileName.split("/")
        folder = fileName[-2]
        image_numb = int(fileName[-1][3:-4])
        for pred in predictions:
            pred_data = pred.json_data()
            categ = pred_data['category_id']
            x,y,w,h = pred_data['bbox']
            x1, x2 = np.clip([x, x+w], a_min=0, a_max=width)
            y1, y2 = np.clip([y, y+h], a_min=0, a_max=height)
            s = pred_data['score']
            image_annotations.append([image_numb,-1,x1, y1, x2-x1, y2-y1, s, 1, categ])

        if len(image_annotations)>0:
            if not folder in self.predictions.keys():
                self.predictions[folder] = np.asarray(image_annotations)
            else:
                self.predictions[folder] = np.concatenate([self.predictions[folder], image_annotations])


    def write_predictions(self, filename):
        mkdir_if_missing(filename)
        for imageName in self.predictions.keys():
            with open(os.path.join(filename,imageName+'.txt'), "w") as file:
                file.write("\n".join([",".join(item) for item in self.predictions[imageName].astype(str)]))

        LOG.info('wrote predictions to %s', filename)
        with zipfile.ZipFile(os.path.join(filename,'predictions')+ '.zip', 'w') as myzip:
            for imageName in self.predictions.keys():
                myzip.write(os.path.join(filename,imageName+'.txt'))
        LOG.info('wrote %s.zip', os.path.join(filename,'predictions')+ '.zip')

    def write_evaluations(self, filename):

        for imageName in self.predictions.keys():
            mkdir_if_missing(filename)
            with open(os.path.join(filename,imageName+".txt"), "w") as file:
                file.write("\n".join(self.predictions[imageName]))

    def stats(self):
        # allgt, alldet, allgt_ign = [], [], []
        # for folder in self.predictions.keys():
        #     allgt.append(self.gt[folder])
        #     alldet.append(self.predictions[folder])
        #     allgt_ign.append(self.gt_ign[folder])
        AP = CalculateDetectionPR_overall2(self.predictions, self.gt, self.gt_ign)
        # AP_obj = {}
        # for obj_attr in range(1,3):
        #     # 1 for Vehicle Category;
        #     # 2 for Vehicle Occlusion;
        #     # 3 for Out-of-view;
        #     AP_obj[obj_attr] = CalculateDetectionPR_obj2(np.asarray(alldet), np.asarray(allgt), np.asarray(allgt_ign), obj_attr);

        AP_time = CalculateDetectionPR_seq2(self.predictions, self.gt, self.gt_ign)
        stats = [AP]
        self.text_labels = ['AP']
        for time, ap_single in AP_time.items():
            stats.append(ap_single)
            self.text_labels.append('AP_'+time)
        data = {
            'stats': stats,
            'text_labels': self.text_labels,
        }

        return data
