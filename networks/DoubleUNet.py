""" Full assembly of the parts to form the complete network """

from .DoubleUNet_parts import *


class DoubleUNet(nn.Module):
    def __init__(self, job_description, trilinear=True):
        super(DoubleUNet, self).__init__()
        self.n_channels = 1
        self.n_classes = 1
        self.n_rec_maps = 1
        self.base_features_e = job_description['base_features']
        self.base_features_e_2 = job_description['base_features']
        self.base_features_d = job_description['base_features']
        self.dropout_rate = job_description['dropout_rate']
        self.trilinear = trilinear

        # 1st UNet
        self.inc = DoubleConv_input(1, self.base_features_e, job_description)
        self.down1 = Down(self.base_features_e, self.base_features_e*2, job_description)
        self.down2 = Down(self.base_features_e*2, self.base_features_e*4, job_description)
        self.down3 = Down(self.base_features_e*4, self.base_features_e*8, job_description)
        self.down4 = Down(self.base_features_e*8, self.base_features_e*16, job_description)
        self.up1_skip = Up(self.base_features_e * 8 + self.base_features_e * 16, self.base_features_d * 8, job_description, self.trilinear)
        self.up2_skip = Up(self.base_features_e * 4 + self.base_features_d * 8, self.base_features_d * 4, job_description, self.trilinear)
        self.up3_skip = Up(self.base_features_e * 2 + self.base_features_d * 4, self.base_features_d * 2, job_description, self.trilinear)
        self.up4_skip = Up(self.base_features_e * 1 + self.base_features_d * 2, self.base_features_d, job_description, self.trilinear)

        self.seg = Out_Seg(self.base_features_d, 1, job_description)
        self.rec = Out_Rec(self.base_features_d, 1, job_description)

        # 2nd UNet
        self.inc_d = DoubleConv_input(1, self.base_features_e_2, job_description)
        self.down1_d = Down(self.base_features_e_2, self.base_features_e_2*2, job_description)
        self.down2_d = Down(self.base_features_e_2*2, self.base_features_e_2*4, job_description)
        self.down3_d = Down(self.base_features_e_2*4, self.base_features_e_2*8, job_description)
        self.down4_d = Down(self.base_features_e_2*8, self.base_features_e_2*16, job_description)
        self.up1_skip_d = Up_3(self.base_features_e_2 * 8 + self.base_features_e_2 * 16 + self.base_features_e * 8, self.base_features_d * 8, job_description, self.trilinear)
        self.up2_skip_d = Up_3(self.base_features_e_2 * 4 + self.base_features_d * 8 + self.base_features_e * 4, self.base_features_d * 4, job_description, self.trilinear)
        self.up3_skip_d = Up_3(self.base_features_e_2 * 2 + self.base_features_d * 4 + self.base_features_e * 2, self.base_features_d * 2, job_description, self.trilinear)
        self.up4_skip_d = Up_3(self.base_features_e_2 * 1 + self.base_features_d * 2 + self.base_features_e * 1, self.base_features_d, job_description, self.trilinear)

        self.seg_d = Out_Seg(self.base_features_d, 1, job_description)
        self.rec_d = Out_Rec(self.base_features_d, 1, job_description)

        self.seg_out = Out_Seg(2, 1, job_description)
        
    def forward(self, image):

        # 1st UNet
        x1 = self.inc(image)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1_skip(x5, x4)
        x = self.up2_skip(x, x3)
        x = self.up3_skip(x, x2)
        x = self.up4_skip(x, x1)

        seg = self.seg(x)
        rec = self.rec(x)

        # 2nd UNet
        image_d = image * seg

        x1_d = self.inc_d(image_d)

        x2_d = self.down1_d(x1_d)
        x3_d = self.down2_d(x2_d)
        x4_d = self.down3_d(x3_d)
        x5_d = self.down4_d(x4_d)

        x_d = self.up1_skip_d(x5_d, x4_d, x4)
        x_d = self.up2_skip_d(x_d, x3_d, x3)
        x_d = self.up3_skip_d(x_d, x2_d, x2)
        x_d = self.up4_skip_d(x_d, x1_d, x1)

        seg_d = self.seg(x_d)
        out = self.seg_out(torch.cat([seg, seg_d], dim=1))

        return out
