import numpy as np
from scipy.linalg import det
import json
import copy
import os
from pathlib import Path
import random
import shutil
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks import LoggerHook
from mmcv.runner.dist_utils import master_only


class UserStop(Exception):
    pass


# Define custom hook to stop process when user uses stop button and to save last checkpoint
try:
    @HOOKS.register_module()
    class CustomHook(Hook):
        # Check at each iter if the training must be stopped
        def __init__(self, stop, output_folder, emitStepProgress):
            self.stop = stop
            self.output_folder = output_folder
            self.emitStepProgress = emitStepProgress

        def after_epoch(self, runner):
            self.emitStepProgress()

        def after_train_iter(self, runner):
            # Check if training must be stopped and save last model
            if self.stop():
                runner.save_checkpoint(self.output_folder, "latest.pth", create_symlink=False)
                raise UserStop
except:
    print("CustomHook already registered")

try:
    @HOOKS.register_module()
    class CustomMlflowLoggerHook(LoggerHook):
        """Class to log metrics and (optionally) a trained model to MLflow.
        It requires `MLflow`_ to be installed.
        Args:
            interval (int): Logging interval (every k iterations). Default: 10.
            ignore_last (bool): Ignore the log of last iterations in each epoch
                if less than `interval`. Default: True.
            reset_flag (bool): Whether to clear the output buffer after logging.
                Default: False.
            by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
        .. _MLflow:
            https://www.mlflow.org/docs/latest/index.html
        """

        def __init__(self,
                     log_metrics,
                     interval=10,
                     ignore_last=True,
                     reset_flag=False,
                     by_epoch=False):
            super(CustomMlflowLoggerHook, self).__init__(interval, ignore_last,
                                                         reset_flag, by_epoch)
            self.log_metrics = log_metrics

        @master_only
        def log(self, runner):
            tags = self.get_loggable_tags(runner)
            if tags:
                self.log_metrics(tags, step=self.get_iter(runner))
except:
    print("MlflowLoggerHook already registered")


def area(pts):
    """
    :param pts: array of coordinates of vertices [[x,y],[x,y]...]
    :return: area of quadrilateral or triangle formed by the points
    """
    n = len(pts)
    if n == 4:
        return area(pts[:3]) + area(np.roll(pts, [-2], axis=[0])[:3])
    if n == 3:
        mat = np.ones((3, 3))
        mat[:, 0] = pts[:, 0]
        mat[:, 1] = pts[:, 1]
        return round(abs(0.5 * det(mat)), 1)


def polygone_to_bbox_xywh(pts):
    """
    :param pts: list of coordinates with xs,ys respectively even,odd indexes
    :return: array of the bounding box xywh
    """
    x = np.min(pts[0::2])
    y = np.min(pts[1::2])
    w = np.max(pts[0::2]) - x
    h = np.max(pts[1::2]) - y
    return [x, y, w, h]


def fill_dict(json_dict, sample, img, gt, id, id_annot):
    json_dict['images'].append({'file_name': img,
                                'segm_file': gt,
                                'height': sample['height'],
                                'width': sample['width'],
                                'id': id})
    for annot in sample['annotations']:
        if 'bbox' in annot:
            annot_to_write = {}
            annot_to_write['iscrowd'] = 0
            annot_to_write['category_id'] = 0
            bbox = annot['bbox']
            x, y, w, h = bbox
            annot_to_write['bbox'] = bbox
            annot_to_write['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
            annot_to_write['area'] = w * h
            annot_to_write['image_id'] = id
            annot_to_write['id'] = id_annot
            json_dict['annotations'].append(annot_to_write)
            id_annot += 1

        elif 'segmentation_poly' in annot:
            poly = annot['segmentation_poly']
            if len(poly):
                if len(poly[0]) > 1:
                    annot_to_write = {}
                    annot_to_write['iscrowd'] = 0
                    annot_to_write['category_id'] = 0
                    annot_to_write['bbox'] = polygone_to_bbox_xywh(poly[0])
                    annot_to_write['segmentation'] = [poly[0].tolist()]
                    annot_to_write['area'] = area(np.array(poly[0].reshape(-1, 2)))
                    annot_to_write['image_id'] = id
                    annot_to_write['id'] = id_annot
                    json_dict['annotations'].append(annot_to_write)
                    id_annot += 1
    return id_annot


def prepare_dataset(ikdata, save_dir, split_ratio):
    paths = {'dataset': os.path.join(save_dir, 'dataset')}

    for a in ['annotations', 'imgs']:
        paths[a] = os.path.join(paths['dataset'], a)
        for b in ['train', 'test']:
            paths[a + '_' + b] = os.path.join(paths['dataset'], a, b)

    for p in paths.values():
        if not (os.path.isdir(p)):
            os.mkdir(p)
    img_list = Path(paths['dataset'] + '/img_list.txt')
    if os.path.isfile(img_list):
        return 0

    print("Preparing dataset...")

    with open(img_list, "w") as f:
        f.write('')
    images = ikdata['images']
    n = len(images)
    train_idx = random.sample(range(n), int(n * split_ratio))
    json_train = {'images': [], 'categories': [{'id': 0, 'name': 'text'}], 'annotations': []}
    json_test = copy.deepcopy(json_train)
    id_annot = 0
    for id, sample in enumerate(images):
        basename = os.path.basename(sample['filename'])
        with open(img_list, 'a') as f:
            f.write(Path(sample['filename']).name + '\n')
        if id in train_idx:
            img = os.path.join(paths['imgs_train'], basename)
            gt = os.path.join(paths['annotations_train'], 'gt_' + os.path.splitext(basename)[0] + '.txt')
            id_annot = fill_dict(json_train, sample, img, gt, id, id_annot)
        else:
            img = os.path.join(paths['imgs_test'], basename)
            gt = os.path.join(paths['annotations_test'], 'gt_' + os.path.splitext(basename)[0] + '.txt')
            id_annot = fill_dict(json_test, sample, img, gt, id, id_annot)

        shutil.copyfile(sample['filename'], img)

        write_annot(sample, gt)
    with open(paths['dataset'] + '/instances_train.json', 'w') as f:
        json.dump(json_train, f)
    with open(paths['dataset'] + '/instances_test.json', 'w') as f:
        json.dump(json_test, f)

    print("Dataset prepared!")


def write_annot(sample, dst_file):
    str_to_write = ''
    for annot in sample['annotations']:
        if 'bbox' in annot:
            x, y, w, h = annot['bbox']
            poly = [x, y, x + w, y, x + w, y + h, x, y + h]
            for number in poly:
                str_to_write += str(int(number)) + ','
        else:
            for poly in annot['segmentation_poly']:
                for number in poly:
                    str_to_write += str(int(number)) + ','
        str_to_write += annot['text']
        str_to_write += '\n'
    with open(dst_file, 'w') as f:
        f.write(str_to_write)
