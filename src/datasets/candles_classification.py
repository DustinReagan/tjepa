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
    dataset = SupervisedCandleDataSet( 'data/binanceus/', freq="1d", training=training, transform=transform)
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

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional

class SupervisedCandleDataSet(Dataset):
    def __init__(self, data_dir: str, freq="1d", sequence_length=448, columns=['open', 'close', 'high', 'low', 'volume'], random_state=42, transform: Optional[Callable] = None, training=True):
        self.sequence_length = sequence_length
        self.columns = columns
        self.random_state = np.random.RandomState(random_state)
        self.transform = transform
        self.training = training
        self.sequences = []  # To store preloaded sequences
        self.labels = []  # To store labels for each sequence

        # Find all relevant data files
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f'-{freq}.feather')]
        if not self.data_files:
            raise ValueError("No data files found in the specified directory.")

        # Load data from files and preprocess
        self.preprocess_data()

    def preprocess_data(self):
        for file_path in self.data_files:
            df = pd.read_feather(file_path)
            if len(df) < self.sequence_length + 1:  # +1 to include the next time-step for class calculation
                continue

            data = df[self.columns].to_numpy()
            # Adjust loop to include next time-step for class calculation
            for start_idx in range(len(df) - self.sequence_length):
                end_idx = start_idx + self.sequence_length
                sequence = data[start_idx:end_idx]
                next_candle = data[end_idx]  # Next time-step after the sequence

                self.sequences.append(sequence)
                self.labels.append(self.calculate_class(sequence[-1][1], next_candle[2]))  # Close of last in sequence and High of next

    def calculate_class(self, last_close, next_high):
        """
        Calculate class based on the percentage change between the last close price and the high price of the next candle.
        """
        return ((next_high - last_close) / last_close) * 100


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        data_sequence = self.sequences[idx]
        label = self.labels[idx]

        # Convert to PyTorch tensor and transpose to match expected shape [num_channels, sequence_length]
        data_tensor = torch.from_numpy(data_sequence).float().transpose(0, 1)

        # Apply transforms if specified and in training mode
        if self.training and self.transform is not None:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label
# Example usage
if __name__ == "__main__":

    dataset,loader, sampler = make_timeseries(transform=None, batch_size=32)

    # Iterate over the DataLoader and process the data (e.g., visualize or feed into a model)
    for i, batch in enumerate(loader):
        print(f"Batch {i}, Shape: {batch.shape}")
        # Add visualization or model feeding code here
        # For simplicity, we're just printing the shape of each batch
        if i == 1:  # Limit to printing information for two batches
            break
    print('done')