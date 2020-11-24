import pytest
from os.path import join
import numpy as np
import torch

from model import MyCNN
from interpret import GradCam, GuidedBackpropReLUModel
from interpret import preprocess_signals, deprocess_signals


@pytest.fixture
def model(const):
    return MyCNN(
        num_channel=const["NUM_CHANNEL"],
        num_class=const["NUM_CLASSES"],
        chunk_size=const["DURATION"]
    )


def test_preprocess_signals():
    pass


def test_deprocess_signals():
    pass


def test_GradCamAndGB(model, const, helpers):

    data = helpers.simulate_ecg_signal(duration=const["DURATION"])
    sample = preprocess_signals(data['signal'])
    grad_cam = GradCam(
        model=model.network,
        feature_module=model.network[:9],
        target_layer_names=["8"]
    )
    cam_mask = grad_cam(sample)
    assert cam_mask.shape == (1, const["NUM_CHANNEL"], const["DURATION"])

    gb_model = GuidedBackpropReLUModel(model=model.network)
    gb = gb_model(sample)
    assert gb.shape == (const["NUM_CHANNEL"], const["DURATION"])
