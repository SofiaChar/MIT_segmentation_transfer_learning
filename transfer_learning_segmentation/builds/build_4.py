import torch

from transfer_learning_segmentation.configs import configs
from transfer_learning_segmentation.models.human_seg_model import HumanSegModel
from transfer_learning_segmentation.models.seg_conv2d_only import SegModelConv2D

build_4_config = {
    'build_name': 'unet_head_4_1',
    'model_class': SegModelConv2D,
    'use_features': False,
    'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

}

build_4_config.update(configs)
