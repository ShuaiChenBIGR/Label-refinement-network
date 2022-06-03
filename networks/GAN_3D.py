""" Full assembly of the parts to form the complete network """
import random
from .GAN_3D_parts import *


class GAN_3D(nn.Module):
    def __init__(self, job_description, trilinear=True, skip=True):
        super(GAN_3D, self).__init__()
        self.n_channels = 1
        self.n_channels_GAN = 2
        self.n_classes = 1
        self.n_rec_maps = 1
        self.base_features = job_description['base_features']
        self.dropout_rate = job_description['dropout_rate']
        self.trilinear = trilinear
        self.skip = skip

        self.inc = DoubleConv_input(self.n_channels, self.base_features, job_description)

        self.down1 = Down(self.base_features, self.base_features*2, job_description)
        self.down2 = Down(self.base_features*2, self.base_features*4, job_description)
        self.down3 = Down(self.base_features*4, self.base_features*8, job_description)
        self.down4 = Down(self.base_features*8, self.base_features*8, job_description)

        if self.skip:
            self.up1_skip = Up(self.base_features*(8+8), self.base_features*4, job_description, self.trilinear)
            self.up2_skip = Up(self.base_features*(4+4), self.base_features*2, job_description, self.trilinear)
            self.up3_skip = Up(self.base_features*(2+2), self.base_features, job_description, self.trilinear)
            self.up4_skip = Up(self.base_features*(1+1), self.base_features, job_description, self.trilinear)
        else:
            self.up1 = Up_no_skip(self.base_features*(8), self.base_features*4, job_description, self.trilinear)
            self.up2 = Up_no_skip(self.base_features*(4), self.base_features*2, job_description, self.trilinear)
            self.up3 = Up_no_skip(self.base_features*(2), self.base_features, job_description, self.trilinear)
            self.up4 = Up_no_skip(self.base_features*(1), self.base_features, job_description, self.trilinear)

        self.seg = Out_Seg(self.base_features, self.n_classes, job_description)
        self.rec = Out_Rec(self.base_features, self.n_rec_maps, job_description)

        # classifier:
        self.inc_d = DoubleConv_input(self.n_channels_GAN, self.base_features, job_description)

        self.down1_d = Down(self.base_features, self.base_features*2, job_description)
        self.down2_d = Down(self.base_features*2, self.base_features*4, job_description)
        self.down3_d = Down(self.base_features*4, self.base_features*8, job_description)
        self.down4_d = Down(self.base_features*8, self.base_features*8, job_description)

        self.classifier_d = Classifier(job_description)

    def forward(self, x, generated_errors, real_error, GD, epoch, job_description):

        if GD == 'G':
            for param in self.inc.parameters():
                param.requires_grad = True
            for param in self.down1.parameters():
                param.requires_grad = True
            for param in self.down2.parameters():
                param.requires_grad = True
            for param in self.down3.parameters():
                param.requires_grad = True
            for param in self.down4.parameters():
                param.requires_grad = True
            for param in self.up1_skip.parameters():
                param.requires_grad = True
            for param in self.up2_skip.parameters():
                param.requires_grad = True
            for param in self.up3_skip.parameters():
                param.requires_grad = True
            for param in self.up4_skip.parameters():
                param.requires_grad = True
            for param in self.seg.parameters():
                param.requires_grad = True
            for param in self.rec.parameters():
                param.requires_grad = True

            for param in self.inc_d.parameters():
                param.requires_grad = False
            for param in self.down1_d.parameters():
                param.requires_grad = False
            for param in self.down2_d.parameters():
                param.requires_grad = False
            for param in self.down3_d.parameters():
                param.requires_grad = False
            for param in self.down4_d.parameters():
                param.requires_grad = False
            for param in self.classifier_d.parameters():
                param.requires_grad = False

        elif GD == "D":
            for param in self.inc.parameters():
                param.requires_grad = False
            for param in self.down1.parameters():
                param.requires_grad = False
            for param in self.down2.parameters():
                param.requires_grad = False
            for param in self.down3.parameters():
                param.requires_grad = False
            for param in self.down4.parameters():
                param.requires_grad = False
            for param in self.up1_skip.parameters():
                param.requires_grad = False
            for param in self.up2_skip.parameters():
                param.requires_grad = False
            for param in self.up3_skip.parameters():
                param.requires_grad = False
            for param in self.up4_skip.parameters():
                param.requires_grad = False
            for param in self.seg.parameters():
                param.requires_grad = False
            for param in self.rec.parameters():
                param.requires_grad = False

            for param in self.inc_d.parameters():
                param.requires_grad = True
            for param in self.down1_d.parameters():
                param.requires_grad = True
            for param in self.down2_d.parameters():
                param.requires_grad = True
            for param in self.down3_d.parameters():
                param.requires_grad = True
            for param in self.down4_d.parameters():
                param.requires_grad = True
            for param in self.classifier_d.parameters():
                param.requires_grad = True

        x1 = self.inc(generated_errors)

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

        rec = self.rec(x_up_4)

        select = random.randint(0, 1)
        # after epoch_G + epoch_D, the generator should fool the discriminator
        if epoch > job_description['epoch_G'] + job_description['epoch_D']:
            select = 1
        # if Real error:
        if select == 0:
            x_d = torch.cat([x, real_error], dim=1)
        # if Fake error:
        elif select == 1:
            x_d = torch.cat([x, rec], dim=1)

        x1_d = self.inc_d(x_d)

        x2_d = self.down1_d(x1_d)
        x3_d = self.down2_d(x2_d)
        x4_d = self.down3_d(x3_d)
        x5_d = self.down4_d(x4_d)

        x5_d_flatten = torch.flatten(x5_d)
        cla = self.classifier_d(x5_d_flatten)

        return rec, cla, select


