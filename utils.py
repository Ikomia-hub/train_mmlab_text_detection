import numpy as np
from scipy.linalg import det
import json
import copy
import os
from pathlib import Path
import random
import shutil
from mmcv.runner.hooks import HOOKS, Hook


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
                runner.save_checkpoint(self.output_folder,"latest.pth",create_symlink=False)
                raise UserStop
except:
    pass

textdet_models = {
    'DB_r18': {
        'config':
            'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
        'ckpt':
            'dbnet/'
            'dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
    },
    'DB_r50': {
        'config':
            'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
        'ckpt':
            'dbnet/'
            'dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20210325-91cef9af.pth'
    },
    'DRRG': {
        'config': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
        'ckpt': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500-1abf4f67.pth'
    },
    'FCE_IC15': {
        'config': 'fcenet/fcenet_r50_fpn_1500e_icdar2015.py',
        'ckpt': 'fcenet/fcenet_r50_fpn_1500e_icdar2015-d435c061.pth'
    },
    'FCE_CTW_DCNv2': {
        'config': 'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py',
        'ckpt': 'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500-05d740bb.pth'
    },
    'MaskRCNN_CTW': {
        'config':
            'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
        'ckpt':
            'maskrcnn/'
            'mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth'
    },
    'MaskRCNN_IC15': {
        'config':
            'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
        'ckpt':
            'maskrcnn/'
            'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
    },
    'MaskRCNN_IC17': {
        'config':
            'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
        'ckpt':
            'maskrcnn/'
            'mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
    },
    'PANet_CTW': {
        'config':
            'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
        'ckpt':
            'panet/'
            'panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
    },
    'PANet_IC15': {
        'config':
            'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
        'ckpt':
            'panet/'
            'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
    },
    'PS_CTW': {
        'config': 'psenet/psenet_r50_fpnf_600e_ctw1500.py',
        'ckpt':
            'psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
    },
    'PS_IC15': {
        'config':
            'psenet/psenet_r50_fpnf_600e_icdar2015.py',
        'ckpt':
            'psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth'
    },
    'TextSnake': {
        'config':
            'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
        'ckpt':
            'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth'
    }
}


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
        poly = annot['segmentation_poly']
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
        for poly in annot['segmentation_poly']:
            for number in poly:
                str_to_write += str(int(number)) + ','
        str_to_write += annot['text']
        str_to_write += '\n'
    with open(dst_file, 'w') as f:
        f.write(str_to_write)
