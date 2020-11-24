# -*- coding: utf-8 -*-
# Author: Hao Chun Chang <changhaochun84@gmail.comm>
#

import os
from os.path import join
import json
import numpy as np
import pandas as pd

import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import pytorch_lightning as pl


class ECGDataModule(pl.LightningDataModule):

    def __init__(self, data_path, label_path, config_path):
        super().__init__()

        self.data_path = data_path
        self.n_channels = 15
        self.config = self._load_config(config_path)
        self.chunk_size = self.config["chunk_size"]
        self.raw_labels = self._extract_labels(label_path)

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return json.load(f)

    def _extract_labels(self, label_path):
        """
        Process labels into one-hot encoding format
        """
        meta = pd.read_csv(label_path)
        label_col = "Reason_for_admission"

        meta = meta.set_index(meta.patient + "/" + meta.record_id)
        meta[label_col] = meta[label_col].fillna("Unknown")
        self.n_classes = len(meta[label_col].unique())
        return pd.get_dummies(meta[label_col])

    def prepare_data(self):
        path = os.path.dirname(self.data_path)
        chunks_file_path = join(path, "data-window{}.npy".format(self.chunk_size))
        labels_file_path = join(path, "labels-window{}.npy".format(self.chunk_size))

        if os.path.exists(chunks_file_path) and os.path.exists(labels_file_path):
            self.data = np.load(chunks_file_path)
            self.labels = np.load(labels_file_path)
        else:
            raw_data = np.load(self.data_path)
            self.data, self.labels = self._split_samples_into_chunks(
                raw_data, self.raw_labels, self.chunk_size
            )
            np.save(chunks_file_path, self.data)
            np.save(labels_file_path, self.labels)

    def _split_samples_into_chunks(self, data, label, chunk_size):
        print("Spliting samples into chunks...")
        chunks = []
        labels = []
        for idx, sample in data.items():
            size = sample.shape[0]
            n_chunks = size // chunk_size
            for i in range(n_chunks):
                chunks.append(sample[i * chunk_size:(i + 1) * chunk_size])
                labels.append(np.array(label.loc[idx, :]))

        chunks = np.array(chunks)
        labels = np.array(labels)
        return chunks, labels

    def setup(self, stage=None):
        transforms = T.Compose([
            T.ToTensor()
        ])
        self.dataset = ECGDataset(self.data, self.labels, transforms)

        # Split testing data
        train_indices, test_indices = self._split_data_indices(
            validation_split=0.1,
            shuffle_dataset=True
        )
        self.test_dataset = self.dataset[test_indices]
        self.dataset = self.dataset[train_indices]
        print("Train/Valid dataset: {}, Test dataset: {}".format(
            len(self.dataset), len(self.test_dataset)
        ))

        # Split train/validation data
        if stage == "fit" or stage is None:
            train_indices, val_indices = self._split_data_indices(
                validation_split=self.config["validation_split"],
                shuffle_dataset=True
            )
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.val_sampler = SubsetRandomSampler(val_indices)

    def _split_data_indices(self, validation_split=.2, shuffle_dataset=True):
        """
        Creating data indices for training and validation splits.
        """
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        if shuffle_dataset:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        return train_indices, val_indices

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            num_workers=3,
            batch_size=self.config["batch_size"],
            sampler=self.train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            num_workers=3,
            batch_size=self.config["batch_size"],
            sampler=self.val_sampler
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=3,
            batch_size=self.config["batch_size"]
        )


class ECGDataset(Dataset):
    def __init__(self, data, label, transforms=None):
        self.data = data
        self.label = label
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'signal': self.data[idx],
            'label': self.label[idx]
        }

        if self.transforms:
            sample['signal'] = self.transforms(sample['signal'])

        return sample
