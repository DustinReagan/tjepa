# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

_GLOBAL_SEED = 0

def make_transforms(noise_level=0.05):
    """
    Create a sequence of transformations for multi-channel time-series data,
    including adding noise and normalization.
    """
    transform_list = [
        
        NormalizeMultiChannelTimeSeries(),
        #AddNoiseToMultiChannelTimeSeries(noise_level=noise_level),
        #GaussianBlur1D(),
    ]
    # Use torch.nn.Sequential to compose the transforms
    transform_sequence = torch.nn.Sequential(*transform_list)
    return transform_sequence

class AddNoiseToMultiChannelTimeSeries(nn.Module):
    """
    Add Gaussian noise to each channel of a multi-channel time series independently.
    """
    def __init__(self, noise_level=0.1):
        super(AddNoiseToMultiChannelTimeSeries, self).__init__()
        self.noise_level = noise_level

    def forward(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_level
        return tensor + noise

class NormalizeMultiChannelTimeSeries(nn.Module):
    """
    Normalize each channel of a multi-channel time series independently
    to have zero mean and unit variance.
    """
    def __init__(self):
        super(NormalizeMultiChannelTimeSeries, self).__init__()

    def forward(self, tensor):
        normalized_tensor = torch.zeros_like(tensor)
        for i in range(tensor.shape[0]):  # Loop over channels
            channel = tensor[i, :]
            mean = torch.mean(channel)
            std = torch.std(channel)
            normalized_tensor[i, :] = (channel - mean) / std
        return normalized_tensor

class GaussianBlur1D(nn.Module):
    def __init__(self, p=0.5, min_rad=1, max_rad=3):
        super(GaussianBlur1D, self).__init__()
        self.p = p
        self.min_rad = min_rad
        self.max_rad = max_rad

    def forward(self, x):
        # Check probability to apply blur
        if torch.rand(1) > self.p:
            return x  # Return original tensor if not applying blur
        
        # Randomly choose a radius for the Gaussian kernel
        rad = torch.randint(low=self.min_rad, high=self.max_rad + 1, size=(1,)).item()
        kernel_size = 2 * rad + 1
        
        # Generate 1D Gaussian kernel
        kernel = torch.arange(kernel_size).float() - rad
        kernel = torch.exp(-0.5 * (kernel / rad).pow(2))
        kernel /= kernel.sum()
        
        # Reshape kernel for convolution
        kernel = kernel.view(1, 1, kernel_size).repeat(x.shape[0], 1, 1)
        
        # Apply padding to maintain sequence length
        padding = rad
        
        # Convolve each channel independently
        x_padded = F.pad(x, (padding, padding), mode='constant', value=0)
        blurred_x = F.conv1d(x_padded.unsqueeze(1), kernel, groups=x.shape[0])
        
        return blurred_x.squeeze(1)