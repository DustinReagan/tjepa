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
    dataset = CandleDataSet( 'data/binanceus/', freq="1d", training=training, transform=transform)
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
    def __init__(self, data_dir: str, freq="1d", sequence_length=448, columns=['open', 'close', 'high', 'low', 'volume'], random_state=42, transform: Optional[Callable] = None, training=True):
        """
        Initializes the dataset to load financial data from multiple files and generate sequences of specified length.
        Each file represents a distinct stock or cryptocurrency.
        :param data_dir: Directory containing data files.
        :param freq: Frequency of the data, used to filter files by naming convention.
        :param sequence_length: Length of each sequence.
        :param columns: List of columns to include in the dataset.
        :param random_state: Seed for the random number generator.
        :param transform: Optional transformation to apply to each sequence.
        :param training: Flag to determine whether to apply transforms (if any).
        """
        self.sequence_length = sequence_length
        self.columns = columns
        self.random_state = np.random.RandomState(random_state)
        self.transform = transform
        self.training = training
        self.sequences = []  # To store preloaded sequences

        # Find all relevant data files
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f'-{freq}.feather')]
        if not self.data_files:
            raise ValueError("No data files found in the specified directory.")

        # Load data from files and preprocess
        self.preprocess_data()

    def preprocess_data(self):
        """
        Preprocesses data from multiple files by loading it into memory,
        ensuring sequences are contained within a single file.
        """
        for file_path in self.data_files:
            df = pd.read_feather(file_path)
            if len(df) < self.sequence_length:
                # Skip files that do not have enough data for at least one sequence
                continue

            data = df[self.columns].to_numpy()
            # Generate all possible sequences for this file and add them to the list
            for start_idx in range(len(df) - self.sequence_length + 1):
                end_idx = start_idx + self.sequence_length
                self.sequences.append(data[start_idx:end_idx])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Generates a single sample of the dataset, consisting of a preloaded sequence of data.
        """
        data_sequence = self.sequences[idx]

        # Convert to PyTorch tensor and transpose to match expected shape [num_channels, sequence_length]
        data_tensor = torch.from_numpy(data_sequence).float().transpose(0, 1)

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