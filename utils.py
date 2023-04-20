import numpy as np
from scipy.linalg import det
import json
import copy
import os
import random
import shutil
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.hooks import LoggerHook
from mmengine.dist import master_only
from typing import Optional, Sequence, Union

DATA_BATCH = Optional[Union[dict, tuple, list]]


class UserStop(Exception):
    pass


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def register_mmlab_modules():
    # Define custom hook to stop process when user uses stop button and to save last checkpoint
    @HOOKS.register_module(force=True)
    class CustomHook(Hook):
        # Check at each iter if the training must be stopped
        def __init__(self, stop, output_folder, emit_step_progress):
            self.stop = stop
            self.output_folder = output_folder
            self.emit_step_progress = emit_step_progress

        def after_epoch(self, runner):
            self.emit_step_progress()

        def _after_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Union[Sequence, dict]] = None,
                        mode: str = 'train') -> None:
            # Check if training must be stopped and save last model
            if self.stop():
                runner.save_checkpoint(self.output_folder, "latest.pth")
                raise UserStop

    @HOOKS.register_module(force=True)
    class CustomLoggerHook(LoggerHook):
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
                     interval=10):
            super(CustomLoggerHook, self).__init__(interval=interval, log_metric_by_epoch=True)
            self.log_metrics = log_metrics

        def after_val_epoch(self,
                            runner,
                            metrics=None) -> None:
            """All subclasses should override this method, if they need any
            operations after each validation epoch.

            Args:
                runner (Runner): The runner of the validation process.
                metrics (Dict[str, float], optional): Evaluation results of all
                    metrics on validation dataset. The keys are the names of the
                    metrics, and the values are corresponding results.
            """
            tag, log_str = runner.log_processor.get_log_after_epoch(
                runner, len(runner.val_dataloader), 'val')
            runner.logger.info(log_str)
            if self.log_metric_by_epoch:
                # when `log_metric_by_epoch` is set to True, it's expected
                # that validation metric can be logged by epoch rather than
                # by iter. At the same time, scalars related to time should
                # still be logged by iter to avoid messy visualized result.
                # see details in PR #278.
                metric_tags = {k: v for k, v in tag.items() if 'time' not in k}
                runner.visualizer.add_scalars(
                    metric_tags, step=runner.epoch, file_path=self.json_log_path)
                self.log_metrics(tag, step=runner.epoch)
            else:
                runner.visualizer.add_scalars(
                    tag, step=runner.iter, file_path=self.json_log_path)
                self.log_metrics(tag, step=runner.iter + 1)

        def after_train_iter(self,
                             runner,
                             batch_idx: int,
                             data_batch=None,
                             outputs=None):
            """Record logs after training iteration.

            Args:
                runner (Runner): The runner of the training process.
                batch_idx (int): The index of the current batch in the train loop.
                data_batch (dict tuple or list, optional): Data from dataloader.
                outputs (dict, optional): Outputs from model.
            """
            # Print experiment name every n iterations.
            if self.every_n_train_iters(
                    runner, self.interval_exp_name) or (self.end_of_epoch(
                runner.train_dataloader, batch_idx)):
                exp_info = f'Exp name: {runner.experiment_name}'
                runner.logger.info(exp_info)
            if self.every_n_inner_iters(batch_idx, self.interval):
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
                  and not self.ignore_last):
                # `runner.max_iters` may not be divisible by `self.interval`. if
                # `self.ignore_last==True`, the log of remaining iterations will
                # be recorded (Epoch [4][1000/1007], the logs of 998-1007
                # iterations will be recorded).
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            else:
                return
            runner.logger.info(log_str)
            runner.visualizer.add_scalars(
                tag, step=runner.iter + 1, file_path=self.json_log_path)
            self.log_metrics(tag, step=runner.iter + 1)


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
        json.dump(json_train, f, cls=NpEncoder)
    with open(paths['dataset'] + '/instances_test.json', 'w') as f:
        json.dump(json_test, f, cls=NpEncoder)

    print("Dataset prepared!")

