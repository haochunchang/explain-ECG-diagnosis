import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl

STEP_SIZE = 3


class MyCNN(pl.LightningModule):

    def __init__(self, num_class):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, STEP_SIZE),
            nn.Conv2d(16, 32, STEP_SIZE, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, STEP_SIZE, padding=1),
            nn.Conv2d(32, 32, STEP_SIZE, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(49056, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_class)
        )

    def forward(self, x):
        prediction = self.network(x)
        return prediction

    def step(self, batch, batch_idx):
        x, y_hat = batch['signal'], batch['label']
        y = self.network(x)
        y_hat = torch.argmax(y_hat, dim=1)
        loss = nn.CrossEntropyLoss()(y, y_hat)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
