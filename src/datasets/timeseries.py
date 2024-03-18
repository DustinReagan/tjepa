# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

import numpy as np
from typing import Callable, Optional

from logging import getLogger

import torch
from torch.utils.data import Dataset

_GLOBAL_SEED = 0
logger = getLogger()


def make_timeseries(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    drop_last=True,
):
    dataset = SinDataSet(training=training, transform=transform)
    logger.info('TimeSeries dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('TimeSeries unsupervised data loader created')

    return dataset, data_loader, dist_sampler

class SinDataSet(Dataset):
    def __init__(self, num_samples=100000, sequence_length=448, num_channels=1, freq_range=(1, 1000), amp_range=(0.5, 100), random_state=None, transform: Optional[Callable] = None, training=True):
        """
        Initializes the dataset with parameters for generating sin curves, potentially across multiple channels.
        :param num_samples: Number of samples in the dataset.
        :param sequence_length: Length of each time-series sequence.
        :param num_channels: Number of channels in each time-series data point.
        :param freq_range: Tuple indicating the range of frequencies for the sin curves.
        :param amp_range: Tuple indicating the range of amplitudes for the sin curves.
        :param random_state: Seed for the random number generator.
        :param transform: A function/transform that takes in a sample and returns a transformed version.
        :param training: Flag to determine whether to apply transforms.
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.random_state = np.random.RandomState(random_state)
        self.transform = transform
        self.training = training
        self.time_steps = np.linspace(0, 2 * np.pi, sequence_length)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generates a single sample of the dataset.
        """
        # Initialize an array to hold the data for all channels
        data = np.zeros((self.num_channels, self.sequence_length))

        # Generate sin curve data for each channel
        for channel in range(self.num_channels):
            frequency = self.random_state.uniform(*self.freq_range)
            amplitude = self.random_state.uniform(*self.amp_range)
            sin_curve = amplitude * np.sin(frequency * self.time_steps)
            data[channel, :] = sin_curve

        # Convert to PyTorch tensor
        data_tensor = torch.from_numpy(data).float()

        # Apply transforms if specified and in training mode
        if self.transform is not None:
            data_tensor = self.transform(data_tensor)

        return data_tensor

# Example usage
if __name__ == "__main__":
    dataset, unsupervised_loader, unsupervised_sampler = make_timeseries(transform=None, batch_size=32)

    # Iterate over the DataLoader and process the data (e.g., visualize or feed into a model)
    for i, batch in enumerate(unsupervised_loader):
        print(f"Batch {i}, Shape: {batch.shape}")
        # Add visualization or model feeding code here
        # For simplicity, we're just printing the shape of each batch
        if i == 1:  # Limit to printing information for two batches
            break
    print('done')