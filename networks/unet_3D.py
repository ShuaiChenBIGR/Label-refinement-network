""" Full assembly of the parts to form the complete network """

from .unet_3D_parts import *


class UNet_3D(nn.Module):
    def __init__(self, job_description, trilinear=True, skip=True, LR=False):
        super(UNet_3D, self).__init__()
        self.LR = LR
        self.n_channels = 1
        if self.LR:
            self.n_channels = 2
        self.n_classes = 1
        self.n_rec_maps = 1
        self.base_features = job_description['base_features']
        self.dropout_rate = job_description['dropout_rate']
        self.trilinear = trilinear
        self.skip = skip

        self.inc = DoubleConv_input(self.n_channels, self.base_features, job_description)

        self.down1 = Down(self.base_features, self.base_features*2, job_description)
        self.down2 = Down(self.base_features*2, self.base_features*4, job_description)
        self.down3 = Down(self.base_features*4, self.base_features*4, job_description)
        self.down4 = Down(self.base_features*4, self.base_features*4, job_description)

        if self.skip:
            self.up1_skip = Up(self.base_features*(4+4), self.base_features*4, job_description, self.trilinear)
            self.up2_skip = Up(self.base_features*(4+4), self.base_features*2, job_description, self.trilinear)
            self.up3_skip = Up(self.base_features*(2+2), self.base_features, job_description, self.trilinear)
            self.up4_skip = Up(self.base_features*(1+1), self.base_features, job_description, self.trilinear)
        else:
            self.up1 = Up_no_skip(self.base_features*(4), self.base_features*4, job_description, self.trilinear)
            self.up2 = Up_no_skip(self.base_features*(4), self.base_features*2, job_description, self.trilinear)
            self.up3 = Up_no_skip(self.base_features*(2), self.base_features, job_description, self.trilinear)
            self.up4 = Up_no_skip(self.base_features*(1), self.base_features, job_description, self.trilinear)

        self.seg = Out_Seg(self.base_features, self.n_classes, job_description)
        self.rec = Out_Rec(self.base_features, self.n_rec_maps, job_description)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.skip:
            x_up_1 = self.up1_skip(x5, x4)
            x_up_2 = self.up2_skip(x_up_1, x3)
            x_up_3 = self.up3_skip(x_up_2, x2)
            x_up_4 = self.up4_skip(x_up_3, x1)
        elif not self.skip:
            x_up_1 = self.up1(x5)
            x_up_2 = self.up2(x_up_1)
            x_up_3 = self.up3(x_up_2)
            x_up_4 = self.up4(x_up_3)

        seg = self.seg(x_up_4)
        rec = self.rec(x_up_4)

        return seg


