import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl

STEP_SIZE = 3


class MyCNN(pl.LightningModule):

    def __init__(self, num_channel, num_class):
        super().__init__()

        self.num_channel = num_channel
        self.network = nn.Sequential(
            nn.Conv1d(self.num_channel, 16, STEP_SIZE),
            nn.Conv1d(16, 32, STEP_SIZE, padding=1),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 32, STEP_SIZE, padding=1),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 32, STEP_SIZE, padding=1),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 64, STEP_SIZE, padding=1),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 64, STEP_SIZE, padding=1),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(4032, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_class)
        )
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        prediction = self.network(x)
        return prediction

    def step(self, batch, batch_idx):
        x, y = batch['signal'], batch['label']
        x = x.squeeze(1).reshape((-1, self.num_channel, 2048))

        y = torch.argmax(y, dim=1)
        y_pred = self.network(x)

        loss = nn.CrossEntropyLoss()(y_pred, y)
        y_pred = torch.argmax(y_pred, dim=1)
        return loss, y_pred, y

    def training_step(self, batch, batch_idx):
        loss, y, y_pred = self.step(batch, batch_idx)

        self.train_acc(y_pred, y)
        self.log('train_accuracy', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_pred = self.step(batch, batch_idx)

        self.valid_acc(y_pred, y)
        self.log('val_accuracy', self.valid_acc, on_step=True, on_epoch=True)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss, y, y_pred = self.step(batch, batch_idx)

        self.test_acc(y_pred, y)
        self.log('test_accuracy', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
