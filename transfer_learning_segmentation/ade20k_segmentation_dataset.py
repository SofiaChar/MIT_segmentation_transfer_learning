import os
import re
from pathlib import Path
import json
import cv2, zlib, base64, io
import io

import torchvision
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cv2
import torch
from skimage.io import imread
from torch.utils import data
import torchvision.transforms.functional as F

from transfer_learning_segmentation.segmentation_dataset import SegmentationDataSet, sort


def construct_dataset(data_dirs, device):
    inputs = []
    targets = []
    for name, data_dir in data_dirs.items():
        # if 'supervisely' in name:
        #     for ds in list(Path(data_dir).iterdir()):
        #         if '.' in str(ds):
        #             continue
        #         inputs += sort(list(Path(os.path.join(ds, 'img')).iterdir()))
        #         targets += sort(list(Path(os.path.join(ds, 'ann')).iterdir()))
        # elif 'crowd_instance' in name:
        #     inputs += sort(list(Path(os.path.join(data_dir, 'Images')).iterdir()))
        #     targets += sort(list(Path(os.path.join(data_dir, 'Human')).iterdir()))
        # else:
        #     curr_inputs = sort(list(Path(os.path.join(data_dir, 'img')).iterdir()))
        #     curr_targets = sort(list(Path(os.path.join(data_dir, 'ann')).iterdir()))
        #     if 'fashion' in name:
        #         curr_inputs = curr_inputs[-200:]
        #         curr_targets = curr_targets[-200:]
        #     inputs += curr_inputs
        #     targets += curr_targets
        if 'images' in name:
            inputs += sort(list(data_dir.iterdir()))
        if 'masks' in name:
            targets += sort(list(data_dir.iterdir()))

    dataset = SegmentationDataSet(inputs=inputs,
                                  targets=targets,
                                  transform=None, device=device)
    return dataset