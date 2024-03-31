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
    dataset = CoinClassificationCandleDataSet('data/binanceus/', sequence_length=sequence_length, freq="1d", training=training, transform=transform, training_split=training_split)
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

class CoinClassificationCandleDataSet(Dataset):
    def __init__(self, data_dir: str, freq="1d", sequence_length=448, columns=['open', 'close', 'high', 'low', 'volume'], training_split=0.8, random_state=42, transform: Optional[Callable] = None, training=True):
        self.sequence_length = sequence_length
        self.columns = columns
        self.random_state = np.random.RandomState(random_state)
        self.transform = transform
        self.training = training
        self.training_split = training_split
        self.sequences = {}  # Changed to store sequences by cryptocurrency
        self.labels = {}  # Changed to store labels by cryptocurrency
        self.label_to_index = {}  # Maps class names to integers
        self.final_sequences = []  # Stores combined sequences for the mode
        self.final_labels = []  # Stores combined labels for the mode
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f'-{freq}.feather')]
        if not self.data_files:
            raise ValueError("No data files found in the specified directory.")

        self.preprocess_data()

    def preprocess_data(self):
        for file_path in self.data_files:
            df = pd.read_feather(file_path)
            if len(df) < self.sequence_length + 1:
                continue

            data = df[self.columns].to_numpy()
            class_label = os.path.basename(file_path).split('-')[0]

            if class_label not in self.label_to_index:
                self.label_to_index[class_label] = len(self.label_to_index)

            if class_label not in self.sequences:
                self.sequences[class_label] = []
                self.labels[class_label] = []

            for start_idx in range(len(df) - self.sequence_length):
                end_idx = start_idx + self.sequence_length
                sequence = data[start_idx:end_idx]
                self.sequences[class_label].append(sequence)
                self.labels[class_label].append(self.label_to_index[class_label])

        # Split and combine sequences and labels based on training mode
        for class_label in self.sequences:
            total_size = len(self.sequences[class_label])
            split_index = int(total_size * self.training_split)
            if self.training:
                self.final_sequences.extend(self.sequences[class_label][:split_index])
                self.final_labels.extend(self.labels[class_label][:split_index])
            else:
                self.final_sequences.extend(self.sequences[class_label][split_index:])
                self.final_labels.extend(self.labels[class_label][split_index:])

    def __len__(self):
        return len(self.final_sequences)

    def __getitem__(self, idx):
        data_sequence = self.final_sequences[idx]
        label_index = self.final_labels[idx]  # Integer index

        data_tensor = torch.from_numpy(data_sequence).float().transpose(0, 1)
        label_tensor = torch.tensor(label_index, dtype=torch.long)

        data_tensor = self.transform(data_tensor)

        return data_tensor, label_tensor

    def get_num_classes(self):
        return len(self.label_to_index)

    def get_class_labels(self):
        print(self.label_to_index)
