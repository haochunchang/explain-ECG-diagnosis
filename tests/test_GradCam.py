from os.path import join
import numpy as np
import torch

from model import MyCNN
from explain import GradCam, GuidedBackpropReLUModel
from explain import preprocess_signals, show_cam_on_image


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
    model = MyCNN(num_channel=15, num_class=15)
    data = simulate_ecg_signal(duration=2048)

    sample = preprocess_signals(data['signal'])
    grad_cam = GradCam(
        model=model.network,
        feature_module=model.network[:9],
        target_layer_names=["8"]
    )
    mask = grad_cam(sample)
    assert mask.shape == (2048,)

    gb_model = GuidedBackpropReLUModel(model=model.network)
    gb = gb_model(sample)
    print(gb, gb.shape)
    assert 1 == 0, "not passing"
    # gb = gb.transpose((1, 2, 0))
    # cam_mask = cv2.merge([mask, mask, mask])
    # cam_gb = deprocess_image(cam_mask*gb)
    # gb = deprocess_image(gb)
