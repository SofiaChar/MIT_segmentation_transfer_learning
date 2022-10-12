from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F


class HumanSegModel(nn.Module):
    def __init__(self, backbone_path,device, *kwargs):
        super(HumanSegModel, self).__init__()

        self.features_extractor = self.set_backbone(backbone_path,device)

        self.bnorm1 = BNormBlock(4, 16, (3, 3), strides=2)
        self.dw_block1 = DWConv(16, 16, 1)
        self.dw_block2 = DWConv(48, 32, 1)
        self.dw_block2_1 = DWConv(32, 32, 1)
        self.dw_block2_2 = DWConv(32, 32, 1)

        self.dw_block3_str = DWConv(32, 32, 2)
        self.dw_block4 = DWConv(64, 32, 1)
        self.dw_block4_1 = DWConv(32, 32, 1)
        self.dw_block4_2 = DWConv(32, 32, 1)

        self.dw_block5_str = DWConv(32, 64, 2)

        self.bottle_1 = DWConv(64, 128, 1)
        self.drop_1 = nn.Dropout(p=0.2)
        self.bottle_2 = DWConv(128, 128, 1)
        self.drop_2 = nn.Dropout(p=0.2)
        self.bottle_3 = DWConv(128, 128, 1)
        self.drop_3 = nn.Dropout(p=0.2)
        self.bottle_4 = DWConv(128, 64, 1)

        self.dw_block6 = DWConv(64, 32, 1)
        self.up1 = nn.Upsample((128, 128))
        self.dw_block7 = DWConv(64, 32, 1)
        self.dw_block7_1 = DWConv(32, 32, 1)
        self.dw_block7_2 = DWConv(64, 32, 1)

        self.dw_block8 = DWConv(32, 32, 1)
        self.up2 = nn.Upsample((256, 256))
        self.dw_block9 = DWConv(64, 32, 1)
        self.dw_block9_1 = DWConv(32, 32, 1)
        self.dw_block9_2 = DWConv(64, 32, 1)

        self.dw_block10 = DWConv(32, 16, 1)
        self.up3 = nn.Upsample((512, 512))
        self.dw_block11 = DWConv(16, 16, 1)

        self.bnorm4 = BNormBlock(16, 1, (3, 3), padding='same')

        self.last_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        out, features = self.features_extractor(inputs)
        x_back = out[:, 12:13, :, :]
        print('features ', type(features.f3))
        print('x max, min', x_back.max(), x_back.min())
        x = torch.cat([inputs, x_back], 1)

        bn1 = self.bnorm1(x)  # ([8, 16, 256, 256])
        dw1 = self.dw_block1(bn1)  # ([8, 16, 256, 256])
        features_3 = nn.Upsample((256, 256))(features['3'])  # ([8, 16, 256, 256])
        conc1 = torch.concat([dw1, features_3], 1)  # ([8, 48, 256, 256])

        dw2 = self.dw_block2(conc1)  # ([8, 32, 256, 256])
        dw2_1 = self.dw_block2_1(dw2)  # ([8, 32, 256, 256])
        skip1 = self.dw_block2_2(dw2_1)  # ([8, 32, 256, 256])

        dw3 = self.dw_block3_str(skip1)  # ([8, 32, 128, 128])
        features_6 = nn.Upsample((128, 128))(features['6'])  # ([8, 32, 128, 128])
        conc2 = torch.concat([dw3, features_6], 1)  # ([8, 64, 128, 128])

        dw4 = self.dw_block4(conc2)  # ([8, 32, 128, 128])
        dw4_1 = self.dw_block4_1(dw4)  # ([8, 32, 128, 128])
        skip2 = self.dw_block4_2(dw4_1)  # ([8, 32, 128, 128])
        dw5 = self.dw_block5_str(skip2)  # ([8, 64, 64, 64])

        # BOTTLENECK
        x = self.bottle_1(dw5)  # ([8, 128, 64, 64])
        x = self.drop_1(x)  # ([8, 128, 64, 64])
        x = self.bottle_2(x)  # ([8, 128, 64, 64])
        x1 = self.drop_2(x)  # ([8, 128, 64, 64])
        x = self.bottle_3(x)  # ([8, 128, 64, 64])
        x = self.drop_3(x)  # ([8, 128, 64, 64])
        x = x1 + x
        x = self.bottle_4(x)  # ([8, 64, 64, 64])
        # BOTTLENECK end

        dw6 = self.dw_block6(x)  # ([8, 64, 64, 64])
        up1 = self.up1(dw6)  # ([8, 64, 128, 128])
        features_9 = nn.Upsample((128, 128))(features['9'])  # ([8, 32, 128, 128])
        conc3 = torch.concat([up1, features_9], 1)  # ([8, 64, 128, 128])

        dw7 = self.dw_block7(conc3)  # ([8, 32, 128, 128])
        dw7_1 = self.dw_block7_1(dw7)  # ([8, 32, 128, 128])
        conc_skip = torch.concat([dw7_1, skip2], 1)  # ([8, 64, 128, 128])
        dw7_2 = self.dw_block7_2(conc_skip)  # ([8, 32, 128, 128])

        # POSSIBLY ANOTHER BLOCK

        dw8 = self.dw_block8(dw7_2)  # ([8, 32, 128, 128])
        up2 = self.up2(dw8)  # ([8, 32, 256, 256])
        features_11 = nn.Upsample((256, 256))(features['11'])  # ([8, 32, 256, 256])
        conc4 = torch.concat([up2, features_11], 1)  # ([8, 64, 256, 256])

        dw9 = self.dw_block9(conc4)  # ([8, 32, 256, 256])
        dw9_1 = self.dw_block9_1(dw9)  # ([8, 32, 256, 256])
        conc_skip_2 = torch.concat([dw9_1, skip1], 1)

        dw9_2 = self.dw_block9_2(conc_skip_2)  # ([8, 32, 256, 256])

        dw10 = self.dw_block10(dw9_2)  # ([8, 16, 256, 256])
        up3 = self.up3(dw10)  # ([8, 16, 512, 512])
        dw11 = self.dw_block11(up3)  # ([8, 16, 512, 512])

        bn4 = self.bnorm4(dw11)  # ([8, 1, 512, 512])

        x = self.last_conv(bn4)  # ([8, 1, 512, 512])
        x = self.sigmoid(x)  # ([8, 1, 512, 512])
        return x, x_back

    @staticmethod
    def set_backbone(path,device):
        model_conv, _ = load_model(path,device)

        for name, param in model_conv.named_parameters():
            if 'conv' in name and 'encoder' in name:
                if '3' in name or '6' in name or '9' in name or '11' in name:
                    continue
            param.requires_grad = False

        return model_conv


class BNormBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, strides=1, padding='valid', dilate=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=strides, dilation=dilate)
        self.b_norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()
        self.add_pad = False
        if strides > 1:
            self.add_pad = True

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.add_pad:
            x = F.pad(x, (0, 1, 0, 1))
        x = self.b_norm(x)
        x = self.relu(x)
        return x


class DWConv(nn.Module):
    def __init__(self, inp, oup, stride, dilate=(1,1)):
        super(DWConv, self).__init__()
        if dilate != (1,1):
            self.dw_conv = nn.Conv2d(inp, inp, 3, stride, groups=inp, bias=False, dilation=dilate, padding='same')
        else:
            self.dw_conv = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.b_n1 = nn.BatchNorm2d(inp)
        self.pw_conv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.b_n2 = nn.BatchNorm2d(oup)

    def forward(self, inputs):
        # dw
        x = self.dw_conv(inputs)
        x = self.b_n1(x)
        x = nn.ReLU(inplace=True)(x)

        # pw
        x = self.pw_conv(x)
        x = self.b_n2(x)
        x = nn.ReLU(inplace=True)(x)
        return x
