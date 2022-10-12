import torch

from transfer_learning_segmentation.configs import configs
from transfer_learning_segmentation.losses import DiceBCELoss
from transfer_learning_segmentation.models.human_seg_conv2d_only import HumanSegModelConv2D
from transfer_learning_segmentation.models.human_seg_model_4_improve_dilate import HumanSegModel4ImproveDilate
from transfer_learning_segmentation.models.human_seg_model_enc_dec_2_conn import HumanSegModelEncDec2Conn
from transfer_learning_segmentation.models.human_seg_model_enc_dec_4_conn import HumanSegModelEncDec4Conn

build_11_config = {
    'build_name': 'human_seg_conv2d_only_dice50',
    'model_class': HumanSegModelConv2D,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 0.0001,
    'criterion': DiceBCELoss(weight=0.5)

}

build_11_config.update(configs)
