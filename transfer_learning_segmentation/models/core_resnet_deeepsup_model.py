import torch

from mit_semseg.models import ModelBuilder, SegmentationModule


def build_backbone():
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet18dilated',
        fc_dim=512,
        weights='ckpt/ade20k-resnet18dilated-ppm_deepsup/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=512,
        num_class=150,
        weights='ckpt/ade20k-resnet18dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    for name, param in segmentation_module.named_parameters():
        param.requires_grad = False
    # segmentation_module.eval()
    return segmentation_module
