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
    sequence_length,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    drop_last=True,
):
    dataset = SinDataSet(training=training, transform=transform, sequence_length=sequence_length)
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
    def __init__(self, num_samples=100000, sequence_length=448, num_channels=1, freq_range=(1, 100), amp_range=(0.5, 100), random_state=None, transform: Optional[Callable] = None, training=True):
        """
        Initializes the dataset with parameters for generating sin curves, potentially across multiple channels, and labels them with their frequency.
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
        Generates a single sample of the dataset and its corresponding frequency label.
        """
        # Initialize an array to hold the data for all channels
        data = np.zeros((self.num_channels, self.sequence_length))
        frequency = self.random_state.uniform(*self.freq_range)  # Same frequency for all channels in a sample

        # Generate sin curve data for each channel
        for channel in range(self.num_channels):
            amplitude = self.random_state.uniform(*self.amp_range)
            sin_curve = amplitude * np.sin(frequency * self.time_steps)
            data[channel, :] = sin_curve

        # Convert to PyTorch tensor
        data_tensor = torch.from_numpy(data).float()

        # Apply transforms if specified and in training mode
        if self.transform is not None and self.training:
            data_tensor = self.transform(data_tensor)

        # Return the data tensor along with the frequency as the label
        return data_tensor, torch.tensor(frequency/self.freq_range[1], dtype=torch.float)

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset, unsupervised_loader, unsupervised_sampler = make_timeseries(transform=None, batch_size=32, sequence_length=448)

    # Get a single batch from the DataLoader
    for batch_data, frequencies in unsupervised_loader:
        print(f"Batch Shape: {batch_data.shape}, Frequencies: {frequencies}")
        plt.figure(figsize=(15, 10))  # Set the figure size for better visibility

        # Assuming batch_data is shaped [batch_size, num_channels, sequence_length]
        # and you want to plot all sequences in the batch
        for i in range(batch_data.size(0)):
            plt.subplot(5, 7, i+1)  # Change subplot grid dimensions as needed based on batch size
            for channel in range(batch_data.size(1)):
                plt.plot(batch_data[i, channel].numpy(), label=f'Channel {channel}')
            plt.title(f'Frequency: {frequencies[i].item():.2f}')
            plt.tight_layout()
            plt.legend()

        plt.show()

        break  # Only plot the first batch