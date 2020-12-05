# -*- coding: utf-8 -*-
# Author: Hao Chun Chang <changhaochun84@gmail.comm>
#
# Entry-point for training, testing, and explaining models

import os
from os.path import join
from argparse import ArgumentParser

import torch
import numpy as np
import pytorch_lightning as pl

from datasets import ECGDataModule
from model import MyCNN
import utils

path = join(os.getcwd(), "data")
data_path = join(path, "data_raw_train.npz")
label_path = join(path, "meta_train.csv")
config_path = join(os.getcwd(), "config.json")


def train(args, dm, net):
    """
    Train the model.

    Args:
        args (:obj:`pd.DataFrame`): Hyperparameters and configurations for model.
        dm (:obj:`datasets.ECGData.ECGDataModule`): The module for loading data.
        net: (:obj:`pytorch_lightning.LightningModule`): The network system instance.
    """
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint

    early_stop_callback = EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        verbose=True,
        mode="max"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.ckpt_path,
        filename="mycnn-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )
    callback_list = [checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer(
        gpus=args.gpus,
        callbacks=callback_list,
        default_root_dir=args.ckpt_path,
        terminate_on_nan=True,
        deterministic=True
    )
    trainer.fit(net, datamodule=dm)


def test(args, dm, net):

    from pytorch_lightning.metrics.functional.classification import confusion_matrix

    model = net.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        num_channel=dm.n_channels,
        num_class=dm.n_classes
    )
    model.eval()

    dm.config["batch_size"] = 1
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer()
    trainer.test(model, datamodule=dm)

    cmat = confusion_matrix(
        torch.stack(model.predictions),
        torch.stack(model.targets),
        num_classes=dm.n_classes
    )

    utils.plot_confusion_matrix(
        matrix=cmat.numpy(),
        classes=dm.raw_Y.columns,
        figure_name="./figures/cmat.jpg"
    )


def explain(args, dm, net):

    from interpret import Explainer

    model = net.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        num_channel=dm.n_channels,
        num_class=dm.n_classes
    )
    dm.config["batch_size"] = 1
    dm.prepare_data()
    dm.setup()

    print("Generating explanation using GradCam...")
    data = next(iter(dm.train_dataloader()))
    sample = utils.preprocess_signals(data["signal"])
    label = data["label"].numpy().argmax()

    GradCamExplainer = Explainer(
        "GradCam",
        model=model.network,
        feature_module=model.network[:9],
        target_layer_names=["8"]
    )
    cam_mask = GradCamExplainer.explain_instance(sample)
    utils.show_cam_on_image(sample=sample, mask=cam_mask, figure_path="./figures/gradcam.jpg")

    print("Generating explanation using LIME...")
    LimeExplainer = Explainer("Lime", model=model)
    explanation = LimeExplainer.explain_instance(data["signal"], labels=[label])
    sample, mask = explanation.get_instance_and_mask(label)
    print(sample.shape, mask.shape, (sample * mask).shape)

    utils.plot_images(
        sample=sample * mask,
        figure_path="./figures/lime_label{}.jpg".format(label)
    )


def main(args):

    pl.seed_everything(args.seed)
    ecg_dm = ECGDataModule(data_path, label_path, config_path)
    net = MyCNN(
        num_channel=ecg_dm.n_channels,
        num_class=ecg_dm.n_classes,
        chunk_size=ecg_dm.chunk_size
    )

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
    parser.add_argument(
        "mode",
        type=str,
        help="Specifying modes: ['train', 'test', 'explain']"
    )
    parser.add_argument(
        "--ckpt_path",
        default="./model_checkpoints",
        help="Checkpoint path for storing models"
    )
    parser.add_argument("--gpus", default=None, help="Numbers of gpus")
    parser.add_argument("--seed", default=56, help="Random seed for all")

    args = parser.parse_args()

    main(args)
