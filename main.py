import os
from os.path import join

import pytorch_lightning as pl

from datasets import ECGDataModule
from model import MyCNN


def main():
    path = join(os.getcwd(), "data")

    data_path = join(path, "data_raw.npz")
    label_path = join(path, "meta.csv")
    config_path = join(os.getcwd(), "config.json")

    ecg_dm = ECGDataModule(data_path, label_path, config_path)
    net = MyCNN(ecg_dm.n_classes)

    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(net, datamodule=ecg_dm)


if __name__ == "__main__":
    main()
