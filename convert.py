import warnings
import os
# Our libs

import cv2
import numpy as np
import coremltools
from torchinfo import summary
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

from transfer_learning_segmentation.models.seg_conv2d_features_add import SegModelConv2DFeatures
from transfer_learning_segmentation.models.segmentation_model_1 import CoreSegModel

warnings.simplefilter(action='ignore', category=FutureWarning)
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json

from torchvision import transforms
from PIL import Image

import coremltools as ct

from coremltools.models.neural_network import quantization_utils
print(ct.__version__)


# def iou(mask1, mask2):
#     intersection = (mask1 * mask2).sum()
#     if intersection == 0:
#         return 0.0
#     union = torch.logical_or(mask1, mask2).to(torch.int).sum()
#     return intersection / union


def iou(mask1, mask2):
    SMOOTH = 1e-6
    a = mask1.sum()
    b = mask2.sum()
    intr = np.logical_and(mask1 > 0.9, mask2 == 1).sum()
    return (intr + SMOOTH) / (a + b - intr + SMOOTH)

def iou_numpy(outputs: np.array, labels: np.array):
    # outputs = outputs.squeeze(1)
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection) / (union)
    return iou

def get_image(path ='/Users/engineer/Dev/Sofia/CSAIL_semantic_seg/sky_originals/1.2.jpeg', npy=False):
    input_image = Image.open(path).copy().convert('RGB')
    # input_image = Image.open("/Users/engineer/Dev/Sofia/segmenter/test_img/img_10.jpg").copy()

    input_image = input_image.resize((512, 512))
    if npy:
        input_image = np.asarray(input_image)[:,:,0:1]
        input_image *=255

    return input_image

def get_image_numpy(path ='/Users/engineer/Dev/Sofia/CSAIL_semantic_seg/sky_originals/1.2.jpeg'):
    input_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image,(512,512))
    if input_image.max() >250:
        input_image = input_image / 255
    return input_image

def convert_torch_model():
    # @register_torch_op
    # def _convolution_mode(context, node):
    #     inputs = _get_inputs(context, node, expected=7)
    #     context.add(mb.conv(
    #         x=inputs[0],
    #         weight=inputs[1],
    #         bias=inputs[2],
    #         strides=inputs[3],
    #         pad_type="same",
    #         dilations=inputs[5],
    #         groups=inputs[6],
    #         name=node.name
    #     )
    #

    @register_torch_op
    def concat(context, node):
        inputs = _get_inputs(context, node)
        values = inputs[0]
        axis = inputs[1].val
        x_pad = mb.concat(values=values, axis=axis)
        context.add(x_pad, node.name)

    path = 'ckpt/checkpoint_best_model.pth'
    # model_class = SegModelConv2DFeatures()
    model = CoreSegModel(SegModelConv2DFeatures, True)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # # Trace the Wrapped Model
    # net_encoder = ModelBuilder.build_encoder(
    #     arch='resnet18dilated',
    #     fc_dim=512,
    #     weights='ckpt/ade20k-resnet18dilated-ppm_deepsup/encoder_epoch_20.pth')
    # net_decoder = ModelBuilder.build_decoder(
    #     arch='ppm_deepsup',
    #     fc_dim=512,
    #     num_class=150,
    #     weights='ckpt/ade20k-resnet18dilated-ppm_deepsup/decoder_epoch_20.pth',
    #     use_softmax=True)
    #
    # crit = torch.nn.NLLLoss(ignore_index=-1)
    # segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    # torch.save(segmentation_module, './ckpt/resnet18dilated-ppm_deepsup.pth')

    print(model)
    model.eval()
    trace = torch.jit.trace(model, torch.rand(1, 3, 512, 512))
    scale = 1 / (0.226 * 255.0)
    bias = [- 0.485 / (0.229), - 0.456 / (0.224), - 0.406 / (0.225)]
    input_batch_pil = get_image()

    mlmodel = ct.convert(
        source='pytorch',
        model=trace,
        inputs=[ct.ImageType(shape=(1, 3, 512, 512), name="image", scale=scale, bias=bias)],
        # inputs = [ct.TensorType(shape=(1, 3, 512, 512))]
    )
    spec = mlmodel.get_spec()

    ct.utils.rename_feature(spec, 'var_1058', 'y')
    model = ct.models.MLModel(spec)
    # # Save the model without new metadata
    # model.save("./mlmodels/resnet18_ppm_deepsup_seg.mlmodel")
    model.save("./mlmodels/resnet18_ppm_deepsup_seg_unet_sky_best_32.mlmodel")

    return model

#
# backbone_mlmodel = convert_torch_model()
# backbone_model_fp8 = quantization_utils.quantize_weights(backbone_mlmodel, nbits=8)
# backbone_model_fp8.save("./mlmodels/resnet18_ppm_deepsup_seg_unet_sky_best_8.mlmodel")

# backbone_model_fp8 = quantization_utils.quantize_weights(backbone_mlmodel, nbits=8)
# # backbone_model_fp8.save("./mlmodels/resnet18_ppm_deepsup_seg_8bit.mlmodel")
backbone_model_fp8 =  ct.models.MLModel("./mlmodels/resnet18_ppm_deepsup_seg_unet_sky_best_8.mlmodel")
# root_data = '/Users/engineer/Dev/datasets/semantic_segmentation/sky_segmentation/ime_sky/data'
# root_truth = '/Users/engineer/Dev/datasets/semantic_segmentation/sky_segmentation/ime_sky/groundtruth'
root_data = './sky_originals'
root_truth = './sky_labels'
# print(sorted(os.listdir(root_data)))
# print(sorted(os.listdir(root_truth)))
bce = nn.BCELoss()
f_bce = 0
f_iou=0
ious = []
for im_p, gt_p in zip(sorted(os.listdir(root_data)),sorted(os.listdir(root_truth))):
    input_batch_pil = get_image(os.path.join(root_data, im_p))
    out = backbone_model_fp8.predict({"image": input_batch_pil})
    gt = get_image_numpy(os.path.join(root_truth, gt_p))
    gt = np.expand_dims(gt,2)

    sky = out['y']
    sky = np.squeeze(sky, axis=0)
    sky = np.transpose(sky, [1,2,0])
    # print(sky.max(), sky.min())
    # sky = np.uint8(sky)
    print('gt', type(gt), gt.shape, gt.max(), gt.min())
    print('sky', type(sky), sky.shape, sky.max(), sky.min())

    sky_thresh = (sky > 0.4).astype('float32')
    ious.append(iou(sky_thresh,gt))
    sky_t = torch.Tensor(sky)
    gt = torch.Tensor(gt)
    f_bce += bce(sky_t,gt)
    # print('IOU ', f_iou)
    # print('curr ',  bce(sky,gt))

    print('BCE ', f_bce)
    print('IOU ', ious[-1])

    to_save = Image.fromarray(np.squeeze(sky_thresh * 255)).convert("L")
    to_save.save(os.path.join('/Users/engineer/Dev/Sofia/CSAIL_semantic_seg','sky_result_8bit',im_p[:-4]+'png'))
    plt.imshow(sky_thresh)
    plt.show()
    # print(ious)

    # print(f_iou.shape)

# print

# iou_sum = np.sum(f_iou)
print('IOUS ', ious)
leny = len(os.listdir(root_data))
print(f'Len {leny}')
print('BCE res ', f_bce/leny)
print('iou res ', np.mean(ious))

# backbone_pred = masks[:,12,:,:]
# backbone_pred = np.squeeze(backbone_pred)
# plt.imshow(backbone_pred* 255)
# plt.show()
# # mlmodel = ct.models.MLModel("./mlmodels/Segmenter_base_8bit.mlmodel")
# backbone_model_fp8 = quantization_utils.quantize_weights(backbone_mlmodel, nbits=8)
# human_model_fp8 = quantization_utils.quantize_weights(backbone_mlmodel, nbits=8)

# model_fp16.save("./mlmodels/Segmenter_base_8bit.mlmodel")
# x16, ml16_features = run_mlmodel_model(mlmodel)
# x32, ml32_features = run_mlmodel_model(mlmodel)

# x, torch_features = run_torch_model()
# print(x1.max(), x1.min())
# print(x2.max(), x2.min())

# print('After, before quntisized',np.mean(np.abs(np.abs(np.array(x16)) - np.abs(np.array(x32)))))
# print('After quntisized, torch',np.mean(np.abs(np.abs(np.array(x16)) - np.abs(np.array(x)))))

