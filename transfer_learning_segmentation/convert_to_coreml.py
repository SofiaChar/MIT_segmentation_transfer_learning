import warnings
from collections import OrderedDict, namedtuple
from PIL import Image
import cv2
import numpy as np
import coremltools
from torchinfo import summary
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb
from segm.model.factory import load_model
from transfer_learning_segmentation.models.core_human_seg import set_backbone
from transfer_learning_segmentation.models.human_seg_model_enc_dec_4_conn import HumanSegModelEncDec4Conn

from transfer_learning_segmentation.models.human_seg_model import HumanSegModel
import matplotlib.pyplot as plt

from transfer_learning_segmentation.models.human_seg_model_4_improve import HumanSegModel4Improve
from transfer_learning_segmentation.models.human_seg_model_4_improve_original import HumanSegModel4ImproveOrig

warnings.simplefilter(action='ignore', category=FutureWarning)
import torchvision.transforms.functional as F

import torch
import torch.nn as nn
import json

from torchvision import transforms
from PIL import Image

import coremltools as ct

from coremltools.models.neural_network import quantization_utils


class WrappedHumanSeg(nn.Module):
    def __init__(self, ):
        super(WrappedHumanSeg, self).__init__()
        self.model = torch.load(
            '/Users/engineer/Dev/Sofia/segmenter/transfer_learning_segmentation/runs/unet_seg_4_3rd_edition/checkpoint_best_model.pth',
            'cpu')
        print(self.model)
        self.model.eval()
        self.human_model = self.reorder_weights()
        self.human_model.eval()
        for n, p in self.human_model.named_parameters():
            print(n)

    def reorder_weights(self):
        model_state_dict = self.model.state_dict()
        backbone_state_dict = OrderedDict()
        human_state_dict = OrderedDict()

        for key, value in model_state_dict.items():
            if 'features_extractor' in key:
                new_key = key.replace('features_extractor.', '')
                backbone_state_dict[new_key] = value
            else:
                new_key = key.replace('human_seg.', '')
                human_state_dict[new_key] = value

        human_model = HumanSegModel4Improve()
        human_model.load_state_dict(human_state_dict, strict=True)
        # backbone = set_backbone(
        #     '/Users/engineer/Dev/Sofia/segmenter/transfer_learning_segmentation/runs/unet_seg_4_3rd_edition/checkpoint_best_model.pth',
        #     False, 'cpu', backbone_state_dict)
        # torch.save(backbone_state_dict,
        #            '/Users/engineer/Dev/Sofia/segmenter/transfer_learning_segmentation/runs/unet_seg_enc_dec_4/backbone_state_dict.pth')
        # torch.save(human_state_dict,
        #            '/Users/engineer/Dev/Sofia/segmenter/transfer_learning_segmentation/runs/unet_seg_enc_dec_4/human_state_dict.pth')
        return human_model

    def forward(self, im, res, features):
        # print('out of summary')
        # res, features = self.backbone(x)
        # im = x[0]
        # res = x[1]
        # features = x[2]
        y = self.human_model(im, res, features)
        print('y.shape', y[0].shape)
        return y


# Wrap the Model to Allow Tracing*
class WrappedSegmenter(nn.Module):
    def __init__(self):
        super(WrappedSegmenter, self).__init__()
        self.model = torch.load(
            '/Users/engineer/Dev/Sofia/segmenter/transfer_learning_segmentation/runs/unet_seg_4_3rd_edition/checkpoint_best_model.pth',
            'cpu')
        print(self.model)
        self.model.eval()
        self.backbone = self.reorder_weights()
        self.backbone.eval()
        # for n, p in self.backbone.named_parameters():
        #     print(n)

    def reorder_weights(self):
        model_state_dict = self.model.state_dict()
        backbone_state_dict = OrderedDict()
        human_state_dict = OrderedDict()

        for key, value in model_state_dict.items():
            if 'features_extractor' in key:
                new_key = key.replace('features_extractor.', '')
                backbone_state_dict[new_key] = value
            else:
                new_key = key.replace('human_seg.', '')
                human_state_dict[new_key] = value

        backbone = set_backbone(
            '/Users/engineer/Dev/Sofia/segmenter/transfer_learning_segmentation/runs/unet_seg_4_3rd_edition/checkpoint_best_model.pth',
            False, 'cpu', backbone_state_dict)
        return backbone

    def forward(self, x):
        # summary(self.model, (1, 3, 512, 512))
        # print('out of summary')
        res, features = self.backbone(x)

        print('res.shape', res.shape)
        # x = res[:, 12:13, :, :]
        x = res
        return x, features


def get_image():
    input_image = Image.open("/Users/engineer/Dev/Sofia/segmenter/test_img/img_8.jpg").copy()
    # input_image = Image.open("/Users/engineer/Dev/Sofia/segmenter/test_img/img_10.jpg").copy()

    input_image = input_image.resize((512, 512))
    return input_image


def run_torch_model(type='segmenter'):
    if type == 'segmenter':
        model = WrappedSegmenter()
    else:
        model = WrappedHumanSeg()

    input_image = get_image()
    input_image = F.pil_to_tensor(input_image).float() / 255
    # input_image.show()
    #
    # preprocess = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((512, 512)),
    #     transforms.Normalize(
    #         mean=[0.5, 0.5, 0.5],
    #         std=[0.5, 0.5, 0.5],
    #     ),
    # ])

    input_tensor = F.normalize(input_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    input_batch = input_tensor.unsqueeze(0)

    print('input_batch.shape', input_batch.shape, input_batch.max(), input_batch.min())
    with torch.no_grad():
        output, features = model(input_batch)
    print('features', type(features.f3))

    x_norm = output + abs(output.min())
    torch_predictions = x_norm / x_norm.max()
    # torch_predictions = torch.threshold(torch_predictions, 0.6, 0)
    torch_predictions = np.squeeze(torch_predictions)
    print('torch_predictions', torch_predictions.shape, torch_predictions.max(), torch_predictions.min())
    x = torch_predictions[12, :, :]
    # print('x', x.shape, x.max(), x.min())
    # plt.imshow(x * 255)
    # plt.show()
    # x = torch_predictions[2, :, :]
    # plt.imshow(x * 255)
    # plt.show()
    return output, features


def run_mlmodel_model(mlmodel):
    input_batch_pil = get_image()
    out = mlmodel.predict({"image": input_batch_pil})

    # return out['y'], {'f3': out['f3'], 'f6': out['f6'], 'f9': out['f9'], 'f11': out['f11']}
    return out['y'], {'f3': out['var_301'], 'f6': out['var_528'], 'f9': out['var_755'], 'f11': out['var_912']}


def segmenter_convert_torch_model():
    # # Trace the Wrapped Model
    input_batch_pil = get_image()
    @register_torch_op
    def concat(context, node):
        inputs = _get_inputs(context, node)
        values = inputs[0]
        axis = inputs[1].val
        x_pad = mb.concat(values=values, axis=axis)
        context.add(x_pad, node.name)
    traceable_model = WrappedSegmenter().eval()
    trace = torch.jit.trace(traceable_model, torch.rand((1, 3, 512, 512)))

    scale = 1 / 255.
    bias = [-1, -1, -1]

    # # Convert the model
    mlmodel = ct.convert(
        source='pytorch',
        model=trace,
        # inputs=[ct.TensorType(name="image", shape=(1, 3, 512, 512))],
        inputs=[ct.ImageType(shape=(1, 3, 512, 512), name="image", scale=scale, bias=bias)],
    )
    # # Save the model without new metadata
    mlmodel.save("./mlmodels/Segmenter_base_for_human.mlmodel")
    out = mlmodel.predict({"image": input_batch_pil})
    # out = mlmodel.predict({"image": np.array(input_batch)})

    print('out KEYS', out.keys())
    out = out['y']
    x_norm = out + abs(out.min())
    torch_predictions = x_norm / x_norm.max()
    # torch_predictions = torch.threshold(torch_predictions, 0.6, 0)
    torch_predictions = np.squeeze(torch_predictions)
    print('torch_predictions', torch_predictions.shape, torch_predictions.max(), torch_predictions.min())
    x = torch_predictions[12, :, :]
    # print('x', x.shape, x.max(), x.min())
    # plt.imshow(x * 255)
    # plt.show()
    # x = torch_predictions[2, :, :]
    # plt.imshow(x * 255)
    # plt.show()
    return mlmodel


def human_convert_torch_model():
    @register_torch_op
    def pad(context, node):
        inputs = _get_inputs(context, node)
        x = inputs[0]
        pad = inputs[1].val
        x_pad = mb.pad(x=x, pad=[pad[0], pad[1], pad[2], pad[3]], mode='constant')
        context.add(x_pad, node.name)

    # @register_torch_op
    # def concat(context, node):
    #     inputs = _get_inputs(context, node)
    #     values = inputs[0]
    #     axis = inputs[1].val
    #     x_pad = mb.concat(values=values, axis=axis)
    #     context.add(x_pad, node.name)

    # # Trace the Wrapped Model
    input_batch_pil = get_image()

    traceable_model = WrappedHumanSeg()
    # tpl = namedtuple('features', ['encoder_9', 'encoder_11', 'decoder_1', 'decoder_2'])
    # intuple = tpl(encoder_9=torch.rand((1, 32, 32, 32)), encoder_11=torch.rand((1, 32, 32, 32)),
    #               decoder_1=torch.rand((1, 32, 32, 32)), decoder_2=torch.rand((1, 32, 32, 32)))
    # trace = torch.jit.trace(traceable_model, {'inputs':torch.rand((1, 3, 512, 512)),'mask': torch.rand((1, 150, 512, 512)), 'features':intuple})

    infeatures = (
        torch.rand((1, 32, 32, 32)), torch.rand((1, 32, 32, 32)), torch.rand((1, 32, 32, 32)),
        torch.rand((1, 32, 32, 32)))

    trace = torch.jit.trace(traceable_model,
                            example_inputs=(
                                torch.rand((1, 3, 512, 512)), torch.rand((1, 150, 512, 512)),
                                torch.rand((1, 128, 32, 32))))

    scale = 1 / 255.
    bias = [-1, -1, -1]
    # example_inputs = torch.Tensor(torch.rand((1, 3, 512, 512)), torch.rand((1, 150, 512, 512)), infeatures)
    # Convert the model
    mlmodel = ct.convert(
        source='pytorch',
        model=trace,
        inputs=[ct.ImageType(shape=(1, 3, 512, 512), name="image", scale=scale, bias=bias),
                ct.TensorType(name="masks", shape=(1, 150, 512, 512)),
                ct.TensorType(name="features", shape=(1, 128, 32, 32))],
        # inputs=[ct.ImageType(shape=(1, 3, 512, 512), name="image", scale=scale, bias=bias)],
    )
    # # # Save the model without new metadata
    mlmodel.save("./mlmodels/human_segmentation_tail.mlmodel")
    # out = mlmodel.predict({"image": input_batch_pil})
    # # out = mlmodel.predict({"image": np.array(input_batch)})
    #
    # print('out KEYS', out.keys())
    # out = out['y']
    # x_norm = out + abs(out.min())
    # torch_predictions = x_norm / x_norm.max()
    # # torch_predictions = torch.threshold(torch_predictions, 0.6, 0)
    # torch_predictions = np.squeeze(torch_predictions)
    # print('torch_predictions', torch_predictions.shape, torch_predictions.max(), torch_predictions.min())
    # x = torch_predictions[12, :, :]
    # print('x', x.shape, x.max(), x.min())
    # plt.imshow(x * 255)
    # plt.show()
    # x = torch_predictions[2, :, :]
    # plt.imshow(x * 255)
    # plt.show()
    return mlmodel


def rename_features_in_mlmodel():
    mlmodel = ct.models.MLModel("./mlmodels/Segmenter_base.mlmodel")
    # get input names
    spec = mlmodel.get_spec()

    ct.utils.rename_feature(spec, 'var_301', 'f3')
    ct.utils.rename_feature(spec, 'var_528', 'f6')
    ct.utils.rename_feature(spec, 'var_755', 'f9')
    ct.utils.rename_feature(spec, 'var_912', 'f11')

    input_names = [inp.name for inp in spec.description.input]
    output_names = [out.name for out in spec.description.output]
    print(input_names)
    print('out:')
    print(output_names)
    model = ct.models.MLModel(spec)
    model.save("./mlmodels/Segmenter_base_features.mlmodel")
    return model


def compare_out_features(ml_features, torch_features):
    # diff1 = ml_features['f3'] - np.array(torch_features.f3)
    # diff2 = ml_features['f6'] - np.array(torch_features.f6)
    # diff3 = ml_features['f9'] - np.array(torch_features.f9)
    # diff4 = ml_features['f11'] - np.array(torch_features.f11)
    # print(ml_features['f3'])
    # print(torch_features.f3)
    # print()
    # print('FFFFF66666')
    # print(ml_features['f6'])
    # print(torch_features.f6)
    # print()
    print('FFFFF9999')
    print(ml_features['f9'].shape)
    print(np.array(torch_features.f11).shape)
    print(np.mean(np.abs(np.abs(ml_features['f9']) - np.abs(np.array(torch_features.f6)))))
    print()

    print('FFFFF1111')
    print(ml_features['f11'].shape)
    print(np.array(torch_features.f9).shape)
    print(np.mean(np.abs(np.abs(ml_features['f11']) - np.abs(np.array(torch_features.f9)))))

    # print(np.allclose(ml_features['f3'], np.array(torch_features.f11)))
    # print(np.allclose(ml_features['f6'], np.array(torch_features.f11)))
    # print(np.allclose(ml_features['f9'], np.array(torch_features.f3)))
    # print(np.allclose(ml_features['f11'], np.array(torch_features.f3)))
    return

backbone_mlmodel = segmenter_convert_torch_model()
human_mlmodel=human_convert_torch_model()

input_batch_pil = get_image()
out = backbone_mlmodel.predict({"image": input_batch_pil})
masks = out['y']
features = out['concat_0']

human_pred = human_mlmodel.predict({"image": input_batch_pil, 'masks':masks, 'features': features})['y']
print('human pred', human_pred.shape)
human_pred = np.squeeze(human_pred)
plt.imshow(human_pred* 255)
plt.show()
backbone_pred = masks[:,12,:,:]
backbone_pred = np.squeeze(backbone_pred)
plt.imshow(backbone_pred* 255)
plt.show()
# mlmodel = ct.models.MLModel("./mlmodels/Segmenter_base_8bit.mlmodel")
backbone_model_fp8 = quantization_utils.quantize_weights(backbone_mlmodel, nbits=8)
human_model_fp8 = quantization_utils.quantize_weights(backbone_mlmodel, nbits=8)

# model_fp16.save("./mlmodels/Segmenter_base_8bit.mlmodel")
# x16, ml16_features = run_mlmodel_model(mlmodel)
# x32, ml32_features = run_mlmodel_model(mlmodel)

# x, torch_features = run_torch_model()
# print(x1.max(), x1.min())
# print(x2.max(), x2.min())

# print('After, before quntisized',np.mean(np.abs(np.abs(np.array(x16)) - np.abs(np.array(x32)))))
# print('After quntisized, torch',np.mean(np.abs(np.abs(np.array(x16)) - np.abs(np.array(x)))))

# out = x16
# x_norm = out + abs(out.min())
# torch_predictions = x_norm / x_norm.max()
# # torch_predictions = torch.threshold(torch_predictions, 0.6, 0)
# torch_predictions = np.squeeze(torch_predictions)
# print('torch_predictions', torch_predictions.shape, torch_predictions.max(), torch_predictions.min())
# x = torch_predictions[12, :, :]
# print('x', x.shape, x.max(), x.min())
# plt.imshow(x * 255)
# plt.show()
# x = torch_predictions[2, :, :]
# plt.imshow(x * 255)
# plt.show()

# compare_out_features(ml_features, torch_features)
# # Add new metadata for preview in Xcode
# labels_json = {
#     "labels": ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow",
#                "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train",
#                "tvOrMonitor"]}
#
# mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
# mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)
#
# mlmodel.save("SegmentationModel_with_metadata.mlmodel")
