from __future__ import print_function, division

from collections import OrderedDict

import torch
import torch.nn as nn
# from torchinfo import summary
import torch.nn.functional as F
import torchvision
# from torchvision.transforms import InterpolationMode
from transfer_learning_segmentation.models.human_seg_model import BNormBlock


# from transfer_learning_segmentation.models.set_segmenter_backbone import SegmenterBackbone, create_extractor


class SegModelConv2DFeatures(nn.Module):
    def __init__(self):
        super(SegModelConv2DFeatures, self).__init__()

        self.bnorm1 = BNormBlock(4, 8, (3, 3), padding=(1, 1))
        self.bnorm2 = BNormBlock(8, 8, (3, 3), padding=(1, 1))
        self.bnorm3 = BNormBlock(8, 16, (3, 3), strides=2)
        self.drop1 = nn.Dropout(0.2)

        self.bnorm4 = BNormBlock(16, 16, (3, 3), padding=(1, 1))
        self.bnorm5 = BNormBlock(16, 16, (3, 3), padding=(1, 1))
        self.bnorm6 = BNormBlock(16, 32, (3, 3), strides=2)
        self.drop2 = nn.Dropout(0.2)

        self.bnorm7 = BNormBlock(32, 32, (3, 3), padding=(1, 1))
        self.bnorm8 = BNormBlock(32, 32, (3, 3), padding=(1, 1))
        self.bnorm9 = BNormBlock(32, 32, (3, 3), strides=2)
        self.drop3 = nn.Dropout(0.2)

        self.bnorm10 = BNormBlock(160, 64, (3, 3), padding=(1, 1))
        self.bnorm11 = BNormBlock(64, 64, (3, 3), padding=(1, 1))
        self.bnorm12 = BNormBlock(64, 64, (3, 3), strides=2)
        self.drop4 = nn.Dropout(0.2)

        self.bnorm13 = BNormBlock(256, 64, (3, 3), padding=(1, 1))

        # BottleNeck
        self.bnorm14 = BNormBlock(128, 64, (3, 3), padding=(2, 2), dilate=2)
        self.drop5 = nn.Dropout(0.3)
        self.bnorm15 = BNormBlock(64, 64, (3, 3), padding=(1, 1))
        self.bnorm16 = BNormBlock(64, 64, (3, 3), padding=(2, 2), dilate=2)
        self.drop6 = nn.Dropout(0.1)
        self.bnorm17 = BNormBlock(64, 64, (3, 3), padding=(1, 1))
        # BottleNeck End

        self.bnorm18 = BNormBlock(64, 64, (3, 3), padding=(1, 1))
        self.bnorm19 = BNormBlock(64, 64, (3, 3), padding=(1, 1))

        self.up1 = nn.Upsample((64, 64))
        self.bnorm20 = BNormBlock(64, 64, (3, 3), padding=(1, 1))
        self.bnorm21 = BNormBlock(64, 64, (3, 3), padding=(1, 1))
        self.bnorm22 = BNormBlock(64, 64, (3, 3), padding=(1, 1))

        self.up2 = nn.Upsample((128, 128))
        self.bnorm23 = BNormBlock(96, 32, (3, 3), padding=(1, 1))
        self.bnorm24 = BNormBlock(32, 32, (3, 3), padding=(1, 1))
        self.bnorm25 = BNormBlock(32, 32, (3, 3), padding=(1, 1))

        self.up3 = nn.Upsample((256, 256))
        self.bnorm26 = BNormBlock(48, 16, (3, 3), padding=(1, 1))
        self.bnorm27 = BNormBlock(16, 16, (3, 3), padding=(1, 1))
        self.bnorm28 = BNormBlock(16, 16, (3, 3), padding=(1, 1))

        self.up4 = nn.Upsample((512, 512))
        self.bnorm29 = BNormBlock(24, 8, (3, 3), padding=(1, 1))
        self.bnorm30 = BNormBlock(8, 8, (3, 3), padding=(1, 1))
        self.bnorm31 = BNormBlock(8, 8, (3, 3), padding=(1, 1))

        self.last_conv = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1, 1), padding=0)
        self.softmax = nn.Sigmoid()

    def forward(self, inputs, mask):
        mask, features = mask[0], mask[1]
        # print('features ', len(features), features[0].shape,features[1].shape,features[2].shape)
        print()
        x_back = mask[:, 2:3, :, :]
        x = torch.cat([inputs, x_back], 1)

        x = self.bnorm1(x)
        skip1 = self.bnorm2(x)
        x = self.bnorm3(skip1)  # 256, , c8
        x = self.drop1(x)

        x = self.bnorm4(x)
        skip2 = self.bnorm5(x)
        x = self.bnorm6(skip2)  # 128, c16
        x = self.drop2(x)

        x = self.bnorm7(x)
        skip3 = self.bnorm8(x)
        x = self.bnorm9(skip3)  # 64, c32
        x = self.drop3(x)

        # # Add features_1
        features_1 = features[0]  # 64, c128
        conc1 = torch.concat([x, features_1], 1)  # 64, c160

        x = self.bnorm10(conc1)
        skip4 = self.bnorm11(x)
        x = self.bnorm12(skip4)  # 32, c64
        x = self.drop4(x)
        print('x.shape ', x.shape)

        features_2 = features[1]  # 64, c256
        print('x.shape ', x.shape)

        # features_2_s = torch.nn.functional.interpolate(features_2, scale_factor=(0.5, 0.5))  # 32, c128
        # features_2_s = downsampling(features_2, (32, 32), scale_factor=(0.5, 0.5))
        features_2_s = torchvision.transforms.Resize((32,32), )(features_2)
        features_2_s = self.bnorm13(features_2_s)

        conc2 = torch.concat([x, features_2_s], 1)  # 32, c192

        # features_2 = features[1]   # 64, c256
        # features_2_s = self.conv_down(features_2)  # 32, c128
        # print(features_2_s.shape, x.shape)
        # conc2 = torch.concat([x, features_2_s], 1)  # 32, c192

        # BottleNeck
        inp = self.bnorm14(conc2)
        x = self.drop5(x)
        x = self.bnorm15(x)
        x = x.clone() + inp
        inp = self.bnorm16(x)
        x = self.drop6(x)
        x = self.bnorm17(x)
        x = x.clone() + inp

        # BottleNeck End

        # features_3 = features.decoder_1
        # x = torch.concat([x, features_3], 1)
        x = self.bnorm18(x)
        x = self.bnorm19(x)

        x = self.up1(x)
        # features_4 = nn.Upsample((64, 64))(features.decoder_2)
        # x = torch.concat([x, skip4, features_4], 1)
        x = self.bnorm20(x)
        x = self.bnorm21(x)
        x = self.bnorm22(x)

        x = self.up2(x)  # 128
        x = torch.concat([x, skip3], 1)
        x = self.bnorm23(x)
        x = self.bnorm24(x)
        x = self.bnorm25(x)

        x = self.up3(x)  # 256
        x = torch.concat([x, skip2], 1)
        x = self.bnorm26(x)
        x = self.bnorm27(x)
        x = self.bnorm28(x)

        x = self.up4(x)  # 512
        x = torch.concat([x, skip1], 1)
        x = self.bnorm29(x)
        x = self.bnorm30(x)
        x = self.bnorm31(x)

        x = self.last_conv(x)
        x = self.softmax(x)
        return x


def downsampling(x, size=None, scale_factor=None, mode='nearest'):
    # define size if user has specified scale_factor
    if size is None: size = (int(scale_factor * x.size(2)), int(scale_factor * x.size(3)))
    # create coordinates
    h = torch.arange(0, size[0]) / (size[0] - 1) * 2 - 1
    w = torch.arange(0, size[1]) / (size[1] - 1) * 2 - 1
    # create grid
    grid = torch.zeros(size[0], size[1], 2)
    grid[:, :, 0] = w.unsqueeze(0).repeat(size[0], 1)
    grid[:, :, 1] = h.unsqueeze(0).repeat(size[1], 1).transpose(0, 1)
    # expand to match batch size
    grid = grid.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
    if x.is_cuda: grid = grid.cuda()
    # do sampling
    return F.grid_sample(x, grid, mode=mode)
