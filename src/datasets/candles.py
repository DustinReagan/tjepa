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
    sequence_length,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    drop_last=True,
):
    dataset = CandleDataSet( '../freqtrade/user_data/data/binanceus/', sequence_length=sequence_length, freq="15m", training=training, transform=transform)
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
        Initializes the dataset for on-demand loading of financial data sequences.
        Each file represents a distinct stock or cryptocurrency.
        """
        self.sequence_length = sequence_length
        self.columns = columns
        self.transform = transform
        self.training = training
        self.data_files = []
        self.file_lengths = []
        self.total_sequences = 0
        self.data_frames = []
        # Find all relevant data files
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f'-{freq}.feather')]
        if not files:
            raise ValueError("No data files found in the specified directory.")
        
        # Preprocess to calculate the number of sequences in each file
        self.preprocess_data(files, random_state)

    def preprocess_data(self, files, random_state):
        """
        Preprocesses data from multiple files without loading it into memory.
        Calculates how many sequences each file can provide.
        """
        np.random.seed(random_state)
        for file_path in files:
            df_length = pd.read_feather(file_path, columns=[self.columns[0]]).shape[0]
            if df_length >= self.sequence_length:
                num_sequences = df_length - self.sequence_length + 1
                #self.data_files.append(file_path)
                self.file_lengths.append(num_sequences)
                self.total_sequences += num_sequences
                self.data_frames.append(pd.read_feather(file_path))

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        """
        Lazily loads and generates a single sample of the dataset.
        """
        # Determine which file and where in the file this sequence is
        file_idx, seq_idx = self.locate_sequence(idx)
        
        # Load the specific sequence
        df = self.data_frames[file_idx]#pd.read_feather(self.data_files[file_idx])
        data = df[self.columns].iloc[seq_idx:seq_idx+self.sequence_length].to_numpy()
        
        # Process the data as required
        data_tensor = torch.from_numpy(data).float().transpose(0, 1)
        if self.training and self.transform:
            data_tensor = self.transform(data_tensor)
        
        return data_tensor

    def locate_sequence(self, idx):
        """
        Finds which file and the index within the file a given sequence index corresponds to.
        """
        cumsum = np.cumsum(self.file_lengths)
        file_idx = np.searchsorted(cumsum, idx+1)
        seq_idx_in_file = idx - (cumsum[file_idx-1] if file_idx > 0 else 0)
        return file_idx, seq_idx_in_file


# Example usage
if __name__ == "__main__":

    dataset, unsupervised_loader, unsupervised_sampler = make_timeseries(transform=None, batch_size=32, sequence_length=448)

    # Iterate over the DataLoader and process the data (e.g., visualize or feed into a model)
    for i, batch in enumerate(unsupervised_loader):
        print(f"Batch {i}, Shape: {batch.shape}")
        # Add visualization or model feeding code here
        # For simplicity, we're just printing the shape of each batch
        if i == 1:  # Limit to printing information for two batches
            break
    print('done')