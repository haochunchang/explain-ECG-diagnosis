# -*- coding: utf-8 -*-
# Author: Hao Chun Chang <changhaochun84@gmail.comm>
#

import os
from os.path import join
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import ECGDataModule
from model import MyCNN

path = join(os.getcwd(), "data")
data_path = join(path, "data_raw.npz")
label_path = join(path, "meta.csv")
config_path = join(os.getcwd(), "config.json")
ckpt_path = join(os.getcwd(), "model_checkpoints")
model_name = "mycnn-epoch=00-val_loss=0.94.ckpt"


def train(args, dm, net):
    """Train the models.

    Args:
        args (:obj:`pd.DataFrame`): Hyperparameters and configurations for model.
        dm (:obj:`datasets.ECGData.ECGDataModule`): The module for loading data.
        net: (:obj:`pytorch_lightning.LightningModule`): The network system instance.
    """
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        verbose=False,
        mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=ckpt_path,
        filename='mycnn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    callback_list = [checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer(
        max_epochs=5,
        gpus=args.gpus,
        callbacks=callback_list,
        default_root_dir=ckpt_path,
        deterministic=True
    )
    trainer.fit(net, datamodule=dm)


def test(args, dm, net):

    model = net.load_from_checkpoint(
        checkpoint_path=join(ckpt_path, model_name),
        num_channel=dm.n_channels,
        num_class=dm.n_classes
    )
    trainer = pl.Trainer(
        gpus=args.gpus,
        default_root_dir=ckpt_path,
        deterministic=True
    )
    result = trainer.test(model, datamodule=dm)
    print(result)


def explain(args, dm, net):
    """
    Explaining model predictions by visualization
    """
    from explain import GradCam, show_cam_on_image, preprocess_signals
    # from explain import GuidedBackpropReLUModel
    dm.prepare_data()
    dm.setup()

    model = net.load_from_checkpoint(
        checkpoint_path=join(ckpt_path, model_name),
        num_channel=dm.n_channels,
        num_class=dm.n_classes
    )
    dataloader = dm.train_dataloader()
    data = next(iter(dataloader))

    sample = preprocess_signals(data['signal'])
    # label = data['label']

    grad_cam = GradCam(
        model=model.network,
        feature_module=model.network[:9],
        target_layer_names=["8"]
    )
    mask = grad_cam(sample)

    show_cam_on_image(sample, mask)


def main(args):

    pl.seed_everything(args.seed)

    ecg_dm = ECGDataModule(data_path, label_path, config_path)
    net = MyCNN(ecg_dm.n_channels, ecg_dm.n_classes)

    if args.mode == "train":
        train(args, ecg_dm, net)
    elif args.mode == "test":
        test(args, ecg_dm, net)
    elif args.mode == "explain":
        explain(args, ecg_dm, net)
    else:
        raise ValueError(
            "Unrecongnized mode. There are 3 modes: train, test and explain."
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Main entry point of this project.")
    parser.add_argument("mode", type=str, help="Specifying modes: ['train', 'test', 'explain']")
    parser.add_argument("--gpus", default=None, help="numbers of gpus")
    parser.add_argument("--seed", default=56, help="random seed for all")

    args = parser.parse_args()

    main(args)
