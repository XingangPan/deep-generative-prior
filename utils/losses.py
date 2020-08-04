import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptLoss(object):

    def __init__(self):
        pass

    def __call__(self, LossNet, fake_img, real_img):
        with torch.no_grad():
            real_feature = LossNet(real_img.detach())
        fake_feature = LossNet(fake_img)
        perceptual_penalty = F.mse_loss(fake_feature, real_feature)
        return perceptual_penalty

    def set_ftr_num(self, ftr_num):
        pass


class DiscriminatorLoss(object):

    def __init__(self, ftr_num=4, data_parallel=False):
        self.data_parallel = data_parallel
        self.ftr_num = ftr_num

    def __call__(self, D, fake_img, real_img):
        if self.data_parallel:
            with torch.no_grad():
                d, real_feature = nn.parallel.data_parallel(
                    D, real_img.detach())
            d, fake_feature = nn.parallel.data_parallel(D, fake_img)
        else:
            with torch.no_grad():
                d, real_feature = D(real_img.detach())
            d, fake_feature = D(fake_img)
        D_penalty = 0
        for i in range(self.ftr_num):
            f_id = -i - 1
            D_penalty = D_penalty + F.l1_loss(fake_feature[f_id],
                                              real_feature[f_id])
        return D_penalty

    def set_ftr_num(self, ftr_num):
        self.ftr_num = ftr_num
