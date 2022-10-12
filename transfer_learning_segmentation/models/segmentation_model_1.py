from __future__ import print_function, division
from collections import OrderedDict
import torch
import torch.nn as nn
# from torchinfo import summary

from mit_semseg.models import ModelBuilder, SegmentationModule


class CoreSegModel(nn.Module):
    def __init__(self, human_seg):
        super(CoreSegModel, self).__init__()
        self.features_extractor = self.build_backbone()
        self.seg_model = human_seg
        # summary(self.features_extractor, (1, 3, 512, 512))

    def forward(self, inputs):
        out = self.features_extractor(inputs)
        result = self.seg_model(inputs, out)
        return result

    def build_backbone(self):
        net_encoder = ModelBuilder.build_encoder(
            arch='resnet18dilated',
            fc_dim=512,
            weights='/mnt_sda/ML/Sofia/mit_segmentation/ckpt/ade20k-resnet18dilated-ppm_deepsup/encoder_epoch_20.pth')
        net_decoder = ModelBuilder.build_decoder(
            arch='ppm_deepsup',
            fc_dim=512,
            num_class=150,
            weights='/mnt_sda/ML/Sofia/mit_segmentation/ckpt/ade20k-resnet18dilated-ppm_deepsup/decoder_epoch_20.pth',
            use_softmax=True)

        crit = torch.nn.NLLLoss(ignore_index=-1)
        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        for name, param in segmentation_module.named_parameters():
            param.requires_grad = False
        # segmentation_module.eval()
        return segmentation_module
