import random
import numpy as np
import torch
from scipy.signal import resample


def rrc(sig, crop_ratio_low, crop_ratio_up):
    sig_length = sig.shape[1]
    crop_length = int(random.uniform(crop_ratio_low, crop_ratio_up) * sig_length)
    if sig_length == crop_length:
        start_point = 0
    elif crop_length == 0:
        return sig
    else:
        start_point = np.random.randint(0, sig_length - crop_length)
    end_point = start_point + crop_length
    sig_crop = sig[:, start_point:end_point]
    sig_crop = resample(sig_crop, sig_length, axis=1)
    return sig_crop


def to(sig, mask_ratio_low, mask_ratio_up):
    num_leads = sig.shape[0]
    sig_length = sig.shape[1]
    mask_length = int(random.uniform(mask_ratio_low, mask_ratio_up) * sig_length)
    mask_start = np.random.randint(0, sig_length - mask_length)
    if mask_length > 0:
        sig[:, mask_start:(mask_start + mask_length)] = np.zeros([num_leads, mask_length])
    return sig


def rrc_to(sig, crop_ratio_low=0.5, crop_ratio_up=1.0, mask_ratio_low=0.0, mask_ratio_up=0.5):
    return to(rrc(sig, crop_ratio_low, crop_ratio_up), mask_ratio_low, mask_ratio_up)


class RandomResizeCropTimeOut(object):
    def __init__(self, params=None):
        if params is None:
            self.params = [0.5, 1.0, 0.0, 0.5]
        else:
            self.params = params

    def __call__(self, x):
        return rrc_to(x, self.params[0], self.params[1], self.params[2], self.params[3])


class ToTensor(object):
    def __init__(self, leads=None):
        self.leads = leads
        pass

    def __call__(self, x):
        if self.leads is None:
            return torch.tensor(x).float()
        else:
            return torch.tensor(x[self.leads, :]).float()
