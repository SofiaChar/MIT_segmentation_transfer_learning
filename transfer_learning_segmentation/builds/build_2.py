import torch

from transfer_learning_segmentation.configs import configs
from transfer_learning_segmentation.models.human_seg_model_extended import HumanSegModelExtended

build_2_config = {
    'build_name': 'unet_head_extented_1',
    'model_class': HumanSegModelExtended,
    'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 0.001,

}

build_2_config.update(configs)
