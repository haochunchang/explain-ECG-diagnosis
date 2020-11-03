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
        self.config = self._load_config(config_path)
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

    def prepare_data(self, chunk_size=2048):
        # called only on 1 GPU
        path = os.path.dirname(self.data_path)
        chunks_file_path = join(path, "data-window{}".format(chunk_size))
        labels_file_path = join(path, "labels-window{}".format(chunk_size))

        if os.path.exists(chunks_file_path) and os.path.exists(labels_file_path):
            self.data = np.load(chunks_file_path + ".npy")
            self.labels = np.load(labels_file_path + ".npy")
        else:
            raw_data = np.load(self.data_path)
            self.data, self.labels = self._split_samples_into_chunks(
                raw_data, self.raw_labels, chunk_size
            )
            np.save(chunks_file_path, self.data)
            np.save(labels_file_path, self.labels)

    def _split_samples_into_chunks(self, data, label, chunk_size):
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
        # called on every GPU
        transforms = T.Compose([
            T.ToTensor()
        ])
        if stage == "fit" or stage is None:
            self.dataset = ECGDataset(self.data, self.labels, transforms)

            train_indices, val_indices = self._split_data_indices(
                validation_split=self.config["validation_split"],
                shuffle_dataset=True,
                random_seed=self.config["random_seed"]
            )
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.val_sampler = SubsetRandomSampler(val_indices)

        if stage == "test" or stage is None:
            pass

    def _split_data_indices(self, validation_split=.2, shuffle_dataset=True, random_seed=56):
        """
        Creating data indices for training and validation splits.
        """
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        return train_indices, val_indices

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            sampler=self.train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            sampler=self.val_sampler
        )

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.config["batch_size"])


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
