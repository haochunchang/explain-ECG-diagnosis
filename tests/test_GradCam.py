from os.path import join
import numpy as np
import torch

from model import MyCNN
from explain import GradCam, GuidedBackpropReLUModel
from explain import preprocess_signals, deprocess_signals
from explain import plot_images, show_cam_on_image


def simulate_ecg_signal(duration=2048):
    """
    Generate faked ECG signals by specifying time duration.
    """
    fake_signal = np.random.rand(1, 15, duration)
    fake_signal = fake_signal.astype(np.float32)
    fake_label = np.zeros((15,))
    fake_label[7] = 1
    return {
        "signal": torch.from_numpy(fake_signal),
        "label": fake_label
    }


def test_GradCam():

    # FIXME: Can be moved to fixture
    num_channel = 15
    duration = 2048
    model = MyCNN(num_channel=num_channel, num_class=15)
    data = simulate_ecg_signal(duration=duration)

    sample = preprocess_signals(data['signal'])
    grad_cam = GradCam(
        model=model.network,
        feature_module=model.network[:9],
        target_layer_names=["8"]
    )
    cam_mask = grad_cam(sample)
    assert cam_mask.shape == (1, num_channel, duration)

    gb_model = GuidedBackpropReLUModel(model=model.network)
    gb = gb_model(sample)
    assert gb.shape == (num_channel, duration)
