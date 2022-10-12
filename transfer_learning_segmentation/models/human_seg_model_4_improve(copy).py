from __future__ import print_function, division
import torch
import torch.nn as nn
from transfer_learning_segmentation.models.human_seg_model import BNormBlock, DWConv


class HumanSegModel4Improve(nn.Module):
    def __init__(self):
        super(HumanSegModel4Improve, self).__init__()

        self.bnorm1 = BNormBlock(4, 16, (3, 3), strides=2)
        self.bnorm1_1 = BNormBlock(16, 16, (3, 3),padding='same')

        self.dw_block1 = DWConv(16, 16, 1)
        self.dw_block2 = DWConv(16, 16, 1)
        self.dw_block3_str = DWConv(16, 16, 2)

        # New block
        self.new_bl_1_1 = DWConv(16, 32, 1)
        self.new_bl_1_2 = DWConv(32, 32, 1)
        self.new_bl_1_3 = DWConv(32, 32, 1)
        self.new_bl_1_str = DWConv(32, 32, 2)
        # New block

        self.dw_block4 = DWConv(64, 64, 1)
        self.dw_block4_1 = DWConv(64, 64, 1)
        self.dw_block4_2 = DWConv(64, 64, 1)
        self.dw_block5_str = DWConv(64, 64, 2)
        self.dw_block5_1 = DWConv(96, 96, 1)

        self.bottle_1 = DWConv(96, 128, 1)
        self.drop_1 = nn.Dropout(p=0.2)
        self.bottle_2 = DWConv(128, 128, 1)
        self.drop_2 = nn.Dropout(p=0.2)
        self.bottle_3 = DWConv(128, 128, 1)
        self.drop_3 = nn.Dropout(p=0.2)
        self.bottle_4 = DWConv(128, 128, 1)

        self.dw_block6 = DWConv(128, 64, 1)
        self.dw_block7 = DWConv(96, 96, 1)
        self.dw_block7_1 = DWConv(96, 64, 1)
        self.dw_block7_2 = DWConv(64, 64, 1)

        self.up1 = nn.Upsample((64, 64))
        self.dw_block7_3 = DWConv(64, 64, 1)

        # New block
        self.new_bl_2_1 = DWConv(128, 128, 1)
        self.new_bl_2_2 = DWConv(128, 64, 1)
        self.new_bl_2_3 = DWConv(64, 64, 1)
        self.new_bl_2_4 = DWConv(64, 64, 1)

        # New block

        self.dw_block8 = DWConv(96, 64, 1)
        self.dw_block8_1 = DWConv(64, 64, 1)
        self.dw_block8_2 = DWConv(64, 32, 1)
        self.up2 = nn.Upsample((128, 128))

        self.dw_block9 = DWConv(32, 32, 1)
        self.dw_block9_1 = DWConv(32, 32, 1)
        self.dw_block9_2 = DWConv(32, 32, 1)

        self.up3 = nn.Upsample((256, 256))
        self.dw_block10 = DWConv(48, 32, 1)
        self.dw_block10_1 = DWConv(32, 32, 1)
        # self.dw_block10_2 = DWConv(32, 16, 1)

        self.up4 = nn.Upsample((512, 512))
        self.dw_block11 = DWConv(32, 16, 1)
        # self.dw_block11_1 = DWConv(16, 16, 1)

        self.bnorm4 = BNormBlock(16, 16, (3, 3), padding='same')
        self.bnorm5 = BNormBlock(16, 16, (3, 3), padding='same')
        self.bnorm6 = BNormBlock(16, 16, (3, 3), padding='same')

        self.last_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1), padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, mask, features):
        x_back = mask[:, 12:13, :, :]
        x = torch.cat([inputs, x_back], 1)

        bn1 = self.bnorm1(x)  # ([8, 16, 256, 256])
        skip1 = self.bnorm1_1(bn1)  # ([8, 16, 256, 256])

        dw1 = self.dw_block1(skip1)  # ([8, 16, 256, 256])
        dw2 = self.dw_block2(dw1)  # ([8, 16, 256, 256])
        dw3 = self.dw_block3_str(dw2)  # ([8, 16, 128, 128])

        #  NEW BLOCK
        n_1_1 = self.new_bl_1_1(dw3)  # ([8, 32, 128, 128])
        n_1_2 = self.new_bl_1_2(n_1_1)  # ([8, 32, 128, 128])
        n_1_3 = self.new_bl_1_3(n_1_2)  # ([8, 32, 128, 128])
        n_1_str = self.new_bl_1_str(n_1_3)  # ([8, 32, 64, 64])
        #  NEW BLOCK

        features_1 = nn.Upsample((64, 64))(features.encoder_9)  # ([8, 32, 64, 64])
        conc1 = torch.concat([n_1_str, features_1], 1)  # ([8, 64, 64, 64])

        dw4 = self.dw_block4(conc1)  # ([8, 64, 64, 64])
        dw4_1 = self.dw_block4_1(dw4)  # ([8, 64, 64, 64])
        skip2 = self.dw_block4_2(dw4_1)  # ([8, 64, 64, 64])
        dw5 = self.dw_block5_str(skip2)  # ([8, 64, 32, 32])

        features_2 = features.encoder_11  # ([8, 32, 32, 32])
        conc2 = torch.concat([dw5, features_2], 1)  # ([8, 96, 32, 32])
        dw5_1 = self.dw_block5_1(conc2)  # ([8, 96, 32, 32])

        # BOTTLENECK
        x = self.bottle_1(dw5_1)  # ([8, 128, 32, 32])
        x = self.drop_1(x)  # ([8, 128, 32, 32])
        x = self.bottle_2(x)  # ([8, 128, 32, 32])
        x1 = self.drop_2(x)  # ([8, 128, 32, 32])
        x = self.bottle_3(x)  # ([8, 128, 32, 32])
        x = self.drop_3(x)  # ([8, 128, 32, 32])
        x = x1 + x
        x = self.bottle_4(x)  # ([8, 128, 32, 32])
        # BOTTLENECK end

        dw6 = self.dw_block6(x)  # ([8, 64, 32, 32])
        features_3 = features.decoder_1  # ([8, 32,  32, 32])
        conc3 = torch.concat([dw6, features_3], 1)  # ([8, 96,  32, 32])
        dw7 = self.dw_block7(conc3)  # ([8, 96,  32, 32])
        dw7_1 = self.dw_block7_1(dw7)  # ([8, 64, 32, 32])
        dw7_2 = self.dw_block7_2(dw7_1)  # ([8, 64, 32, 32])

        up1 = self.up1(dw7_2)  # ([8, 64, 64, 64])
        dw7_3 = self.dw_block7_3(up1)  # ([8, 64,  64, 64])
        conc_skip = torch.concat([dw7_3, skip2], 1)  # ([8, 128, 64, 64])

        #  NEW BLOCK
        n_2_1 = self.new_bl_2_1(conc_skip)  # ([8, 128, 64, 64])
        n_2_2 = self.new_bl_2_2(n_2_1)  # ([8, 64, 64, 64])
        n_2_3 = self.new_bl_2_3(n_2_2)  # ([8, 64, 64, 64])
        n_2_4 = self.new_bl_2_4(n_2_3)  # ([8, 64, 64, 64])
        #  NEW BLOCK

        features_4 = nn.Upsample((64, 64))(features.decoder_2)  # ([8, 32, 64, 64])
        conc4 = torch.concat([n_2_4, features_4], 1)  # ([8, 96, 64, 64])
        dw8 = self.dw_block8(conc4)  # ([8, 64, 64, 64])
        dw8_1 = self.dw_block8_1(dw8)  # ([8, 64, 64, 64])
        dw8_2 = self.dw_block8_2(dw8_1)  # ([8, 32, 64, 64])

        up2 = self.up2(dw8_2)  # ([8, 32, 128, 128])
        dw9 = self.dw_block9(up2)  # ([8, 32, 128, 128])
        dw9_1 = self.dw_block9_1(dw9)  # ([8, 32,128, 128])
        dw9_2 = self.dw_block9_2(dw9_1)  # ([8, 32, 128, 128])

        up3 = self.up3(dw9_2)  # ([8, 32, 256, 256])
        conc_skip_2 = torch.concat([up3, skip1], 1)  # ([8, 48, 256, 256])
        dw10 = self.dw_block10(conc_skip_2)  # ([8, 32, 256, 256])
        dw10_1 = self.dw_block10_1(dw10)  # ([8, 32, 256, 256])

        up4 = self.up4(dw10_1)  # ([8, 16, 512, 512])
        dw11 = self.dw_block11(up4)  # ([8, 16, 512, 512])

        bn4 = self.bnorm4(dw11)  # ([8, 16, 512, 512])
        bn5 = self.bnorm5(bn4)  # ([8, 16, 512, 512])
        bn6 = self.bnorm6(bn5)  # ([8, 1, 512, 512])

        x = self.last_conv(bn6)  # ([8, 1, 512, 512])
        x = self.sigmoid(x)  # ([8, 1, 512, 512])
        return x, x_back
