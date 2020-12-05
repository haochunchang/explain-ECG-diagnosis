# -*- coding: utf-8 -*-
# Author: Hao Chun Chang <changhaochun84@gmail.comm>
#
# **One-time script**
# Data preprocessing prior to entering main.py

from os.path import join
import pandas as pd
import numpy as np


def train_test_split(data_path, test_size=0.25):
    x = np.load(join(data_path, "data_raw.npz"))
    y = pd.read_csv(join(data_path, "meta.csv"))
    y = y.set_index(y.patient + "/" + y.record_id)

    y_test = y.sample(frac=test_size, axis=0)
    y_train = y.loc[~y.index.isin(y_test.index)]

    x_train = {}
    x_test = {}
    for fname, sample in x.items():
        if fname in y_train.index:
            x_train[fname] = sample
        else:
            x_test[fname] = sample

    np.savez_compressed(join(data_path, "data_raw_train.npz"), **x_train)
    np.savez_compressed(join(data_path, "data_raw_test.npz"), **x_test)
    y_train.to_csv(join(data_path, "meta_train.csv"), index=False)
    y_test.to_csv(join(data_path, "meta_test.csv"), index=False)


if __name__ == "__main__":
    data_path = "./data/"
    train_test_split(data_path)
