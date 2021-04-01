"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compare_psnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred, data_range=0.5)


def compare_ssim(gt, pred):
    return structural_similarity(gt, pred)


def allowed_losses():
    return loss_dict.keys()


def define_loss(loss_name, *args):
    if loss_name not in allowed_losses():
        raise NotImplementedError(
            'Loss functions {} is not yet implemented'.format(loss_name))
    else:
        return loss_dict[loss_name](*args)


class MSE_loss(nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, prediction, gt):
        # err = prediction[:,0:1] - gt
        err = gt - prediction
        mse_loss = torch.mean((err)**2)
        return mse_loss


class Multi_MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g1, g2, f1, f2):
        error = g1 - g2 - (f1 - f2)
        loss = torch.mean(error**2)
        return loss


loss_dict = {"Multimse": Multi_MSE_Loss, 'mse': MSE_loss}
