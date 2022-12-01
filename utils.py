import numpy as np
from scipy.linalg import det
import json
import copy
import os
from pathlib import Path
import random
import shutil
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.hooks import LoggerHook
from mmengine.dist import master_only
from typing import Dict, Optional, Sequence, Union

DATA_BATCH = Optional[Union[dict, tuple, list]]


class UserStop(Exception):
    pass


def register_mmlab_modules():
    # Define custom hook to stop process when user uses stop button and to save last checkpoint
    @HOOKS.register_module(force=True)
    class CustomHook(Hook):
        # Check at each iter if the training must be stopped
        def __init__(self, stop, output_folder, emitStepProgress):
            self.stop = stop
            self.output_folder = output_folder
            self.emitStepProgress = emitStepProgress

        def after_epoch(self, runner):
            self.emitStepProgress()

        def _after_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Union[Sequence, dict]] = None,
                        mode: str = 'train') -> None:
            # Check if training must be stopped and save last model
            if self.stop():
                runner.save_checkpoint(self.output_folder, "latest.pth", create_symlink=False)
                raise UserStop

    @HOOKS.register_module(force=True)
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
                     by_epoch=False):
            super(CustomMlflowLoggerHook, self).__init__(interval=interval, ignore_last=ignore_last,
                                                         log_metric_by_epoch=by_epoch)
            self.log_metrics = log_metrics

        @master_only
        def log(self, runner):
            tags = self.get_loggable_tags(runner)
            if tags:
                self.log_metrics(tags, step=self.get_iter(runner))


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


def fill_dict(json_dict, sample, img):
    instances = []
    for annot in sample['annotations']:
        if 'bbox' in annot:
            annot_to_write = {}
            annot_to_write['ignore'] = False
            annot_to_write['bbox_label'] = 0
            bbox = [int(c) for c in annot['bbox']]
            x, y, w, h = bbox
            annot_to_write['bbox'] = bbox
            annot_to_write['polygon'] = [x, y, x + w, y, x + w, y + h, x, y + h]
            instances.append(annot_to_write)

        elif 'segmentation_poly' in annot:
            poly = annot['segmentation_poly']
            if len(poly):
                poly = [int(c) for c in poly[0]]
                if len(poly) > 1:
                    annot_to_write = {}
                    annot_to_write['ignore'] = 0
                    annot_to_write['bbox_label'] = 0
                    annot_to_write['bbox'] = polygone_to_bbox_xywh(poly)
                    annot_to_write['polygon'] = poly.tolist() if isinstance(poly, np.ndarray) else poly
                    instances.append(annot_to_write)
    json_dict['data_list'].append({'img_path': img,
                                   'height': sample['height'],
                                   'width': sample['width'],
                                   'instances': instances})


def prepare_dataset(ikdata, save_dir, split_ratio):
    paths = {'dataset': os.path.join(save_dir, 'dataset')}
    dataset_dir = os.path.join(save_dir, 'dataset')
    imgs_dir = os.path.join(dataset_dir, 'images')
    for dire in [dataset_dir, imgs_dir]:
        if not (os.path.isdir(dire)):
            os.mkdir(dire)
        else:
            # delete files already in these directories
            for filename in os.listdir(dire):
                file_path = os.path.join(dire, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    for p in paths.values():
        if not (os.path.isdir(p)):
            os.mkdir(p)

    print("Preparing dataset...")

    images = ikdata['images']
    n = len(images)
    train_idx = random.sample(range(n), int(n * split_ratio))
    json_train = \
        {
            "metainfo":
                {
                    "dataset_type": "TextDetDataset",
                    "task_name": "textdet",
                    "category": [{"id": 0, "name": "text"}]
                },
            'data_list': []
        }
    json_test = copy.deepcopy(json_train)
    for id, sample in enumerate(images):
        basename = os.path.basename(sample['filename'])
        img = os.path.join(imgs_dir, basename)
        if id in train_idx:
            current_json = json_train
        else:
            current_json = json_test
        fill_dict(current_json, sample, img)

        shutil.copyfile(sample['filename'], img)
    with open(paths['dataset'] + '/instances_train.json', 'w') as f:
        json.dump(json_train, f)
    with open(paths['dataset'] + '/instances_test.json', 'w') as f:
        json.dump(json_test, f)

    print("Dataset prepared!")

