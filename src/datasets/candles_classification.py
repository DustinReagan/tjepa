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
    training_split=0.8  # New parameter indicating training data proportion
):
    dataset = RedGreenCandleDataSet('data/binanceus/', sequence_length=sequence_length, freq="15m", training=training, transform=transform, training_split=training_split)
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

class RedGreenCandleDataSet(Dataset):
    def __init__(self, data_dir: str, freq="15m", sequence_length=448, columns=['open', 'close', 'high', 'low', 'volume'], training_split=0.8, random_state=42, transform: Optional[Callable] = None, training=True):
        self.sequence_length = sequence_length
        self.columns = columns
        self.random_state = np.random.RandomState(random_state)
        self.transform = transform
        self.training = training
        self.training_split = training_split
        self.final_sequences = []  # Stores combined sequences for the mode
        self.final_labels = []  # Stores combined labels for the mode
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f'-{freq}.feather') and f.startswith('BTC')]
        if not self.data_files:
            raise ValueError("No data files found in the specified directory.")

        self.preprocess_data()

    def preprocess_data(self):
        sequences_by_coin = {}  # Temporarily stores sequences by coin
        labels_by_coin = {}  # Temporarily stores labels by coin

        for file_path in self.data_files:
            df = pd.read_feather(file_path)
            if len(df) < self.sequence_length + 1:
                continue

            coin_name = os.path.basename(file_path).split('-')[0]
            if coin_name not in sequences_by_coin:
                sequences_by_coin[coin_name] = []
                labels_by_coin[coin_name] = []

            data = df[self.columns].to_numpy()

            for start_idx in range(len(df) - self.sequence_length):
                end_idx = start_idx + self.sequence_length
                sequence = data[start_idx:end_idx]
                next_candle = data[end_idx]  # Next time-step after the sequence
                label = self.calculate_class(next_candle[0], next_candle[1], next_candle[2], next_candle[3])
                
                sequences_by_coin[coin_name].append(sequence)
                labels_by_coin[coin_name].append(label)

        # Sequentially split and combine sequences and labels for the specified mode
        for coin_name in sequences_by_coin:
            total_size = len(sequences_by_coin[coin_name])
            split_index = int(total_size * self.training_split)
            sequences = sequences_by_coin[coin_name]
            labels = labels_by_coin[coin_name]

            if self.training:
                self.final_sequences.extend(sequences[:split_index])
                self.final_labels.extend(labels[:split_index])
            else:
                self.final_sequences.extend(sequences[split_index:])
                self.final_labels.extend(labels[split_index:])

    def calculate_class(self, open_price, close_price, high_price, low_price):
        """
        Determine the classification of a candle based on open, close, high, and low prices.
        A candle is considered:
        - 'Green' (bullish) if close > open
        - 'Red' (bearish) if close < open
        - 'Doji' (neutral) if the body is very small compared to the total range, indicating indecision.
        """
        body = abs(close_price - open_price)
        total_range = high_price - low_price
        epsilon_percentage = 0.05  # Threshold for body to total range ratio to consider as doji, adjust as needed
        
        if total_range == 0:  # Avoid division by zero, very rare case where high, low, open, and close are the same
            return 2  # Neutral/doji
        
        body_to_range_ratio = body / total_range
        if body_to_range_ratio <= epsilon_percentage:
            return 2  # Neutral/doji
        elif close_price > open_price:
            return 0  # Green
        else:
            return 1  # Red

    def __len__(self):
        return len(self.final_sequences)

    def __getitem__(self, idx):
        data_sequence = self.final_sequences[idx]
        label = self.final_labels[idx]

        data_tensor = torch.from_numpy(data_sequence).float().transpose(0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_tensor
    def get_num_classes(self):
        return 3
    def get_class_labels(self):
        return {0: "Green", 1: "Red", 2: "Doji"}
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