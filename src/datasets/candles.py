# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

import pandas as pd
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
    dataset = CandleDataSet( 'data/binanceus/BTC_USDT-1d.feather', training=training, transform=transform)
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

class CandleDataSet(Dataset):
    def __init__(self, file_path: str, sequence_length=448, columns=['open', 'close', 'high', 'low', 'volume'], random_state=None, transform: Optional[Callable] = None, training=True):
        """
        Initializes the dataset to load financial data and generate sequences of specified length.
        :param file_path: Path to the data file.
        :param sequence_length: Length of each sequence.
        :param columns: List of columns to include in the dataset.
        :param random_state: Seed for the random number generator.
        :param transform: Optional transformation to apply to each sequence.
        :param training: Flag to determine whether to apply transforms (if any).
        """
        df = pd.read_feather(file_path)
        self.data = df[columns].to_numpy()

        self.sequence_length = sequence_length
        self.num_channels = len(columns)
        self.random_state = np.random.RandomState(random_state)
        self.transform = transform
        self.training = training

        # Ensure that there is enough data for at least one sequence
        if len(df) < sequence_length:
            raise ValueError("Data length is shorter than the requested sequence length.")

        # Calculate number of samples: each step moves one time point forward
        self.num_samples = len(df) - sequence_length + 1

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generates a single sample of the dataset, consisting of a sequence of data.
        """
        # Extract the sequence from the data
        start_idx = idx
        end_idx = idx + self.sequence_length
        data_sequence = self.data[start_idx:end_idx, :]

        # Convert to PyTorch tensor
        data_tensor = torch.from_numpy(data_sequence).float().transpose(0, 1)  # Shape: [num_channels, sequence_length]

        # Apply transforms if specified and in training mode
        if self.training and self.transform is not None:
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