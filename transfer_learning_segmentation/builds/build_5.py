import torch

from transfer_learning_segmentation.configs import configs
from transfer_learning_segmentation.models.human_seg_model import HumanSegModel
from transfer_learning_segmentation.models.seg_conv2d_features_add import SegModelConv2DFeatures
from transfer_learning_segmentation.models.seg_conv2d_only import SegModelConv2D

build_5_config = {
    'build_name': 'unet_head_5',
    'model_class': SegModelConv2DFeatures,
    'use_features': True,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

}

build_5_config.update(configs)
