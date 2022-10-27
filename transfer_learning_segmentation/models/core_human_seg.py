from __future__ import print_function, division
from collections import OrderedDict
import torch
import torch.nn as nn
# from torchinfo import summary


class CoreHumanSegModel(nn.Module):
    def __init__(self,  human_seg, features_extractor):
        super(CoreHumanSegModel, self).__init__()
        self.features_extractor = features_extractor
        self.human_seg = human_seg
        # summary(self.features_extractor, (1, 3, 512, 512))

    def forward(self, inputs):
        out, features = self.features_extractor(inputs)
        result = self.human_seg(inputs, out, features)
        return result


def create_seg_model(hum_seg_model_class, path, innit, device):
    human_seg = hum_seg_model_class()
    # human_seg.load_state_dict()
    # backbone_state_dict = None
    # if not innit:
    #     # backbone_state_dict, human_state_dict = reorder_weights(path)
    #     human_seg.load_state_dict(human_state_dict, strict=True)
    # features_extractor = set_backbone(path, innit, device, backbone_state_dict)
    return human_seg


def reorder_weights(path):
    model_state_dict = torch.load(path).state_dict()
    backbone_state_dict = OrderedDict()
    human_state_dict = OrderedDict()

    for key, value in model_state_dict.items():
        if 'features_extractor' in key:
            new_key = key.replace('features_extractor.', '')
            backbone_state_dict[new_key] = value
        else:
            new_key = key.replace('human_seg.', '')
            human_state_dict[new_key] = value
    return backbone_state_dict, human_state_dict


def set_backbone(path, innit, device, state_dict):
    if innit:
        backbone_model, _ = load_model(path, device, strict=False)
    else:
        backbone_model = load_model_from_state_dict(path, state_dict, device, strict=True)
    for name, param in backbone_model.named_parameters():
        if 'conv' in name and 'encoder' in name:
            if '9' in name or '11' in name:
                continue
        if 'conv' in name and 'decoder' in name:
            continue
        param.requires_grad = False
    return backbone_model
