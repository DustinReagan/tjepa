# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import torch
import torch.nn as nn

_GLOBAL_SEED = 0

def make_transforms(noise_level=0.1):
    """
    Create a sequence of transformations for multi-channel time-series data,
    including adding noise and normalization.
    """
    transform_list = [
        AddNoiseToMultiChannelTimeSeries(noise_level=noise_level),
        NormalizeMultiChannelTimeSeries()
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
