""" Full assembly of the parts to form the complete network """

from .Adv_parts import *
import random


class Adv(nn.Module):
    def __init__(self, job_description, trilinear=True, testing=False):
        super(Adv, self).__init__()
        self.n_channels = 1
        self.n_classes = 1
        self.n_rec_maps = 1
        self.base_features_e = job_description['base_features']
        self.base_features_d = job_description['base_features']
        self.dropout_rate = job_description['dropout_rate']
        self.trilinear = trilinear
        self.testing = testing
        self.job_description = job_description

        # Unet
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

        # Discriminator
        self.inc_d = DoubleConv_input(2, self.base_features_e, job_description)
        self.down1_d = Down(self.base_features_e, self.base_features_e*2, job_description)
        self.down2_d = Down(self.base_features_e*2, self.base_features_e*4, job_description)
        self.down3_d = Down(self.base_features_e*4, self.base_features_e*8, job_description)
        self.down4_d = Down(self.base_features_e*8, self.base_features_e*1, job_description)
        self.classifier_d = Classifier(job_description)
        
    def forward(self, image, gt):

        # UNet
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

        selection = random.randint(0, 1)

        if self.testing:
            selection = 0

        if selection == 0:
            input_d = seg
        elif selection == 1:
            input_d = gt[:, :]

        input_d = torch.cat([input_d, image], dim=1)
        x1_c = self.inc_d(input_d)
        x2_c = self.down1_d(x1_c)
        x3_c = self.down2_d(x2_c)
        x4_c = self.down3_d(x3_c)
        x5_c = self.down4_d(x4_c)

        x5_flatten = torch.flatten(x5_c, 1)
        classifier = self.classifier_d(x5_flatten)

        return classifier, selection, seg
