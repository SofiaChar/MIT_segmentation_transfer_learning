from __future__ import print_function, division
import torch.nn as nn


class SegmenterBackbone(nn.Module):
    def __init__(self, backbone_path, device, *kwargs):
        super(SegmenterBackbone, self).__init__()

        self.features_extractor = self.set_backbone(backbone_path, device)

    def forward(self, inputs):
        out, features = self.features_extractor(inputs)
        return out, features

    @staticmethod
    def set_backbone(path, device):
        model_conv, _ = load_model(path, device)

        for name, param in model_conv.named_parameters():
            if 'conv' in name and 'encoder' in name:
                if '3' in name or '6' in name or '9' in name or '11' in name:
                    continue
            param.requires_grad = False
        return model_conv


def create_extractor(path, device):
    backbone = SegmenterBackbone(path, device)
    return backbone
