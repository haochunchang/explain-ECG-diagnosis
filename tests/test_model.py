import pytest
import torch

from model import MyCNN


@pytest.fixture
def model(const):
    return MyCNN(
        num_channel=const["NUM_CHANNEL"],
        num_class=const["NUM_CLASSES"],
        chunk_size=const["DURATION"]
    )


def test_MyCNN_forward(model, const):

    dummy = torch.rand(1, const["NUM_CHANNEL"], const["DURATION"])
    output = model.forward(dummy)
    assert output.shape == (1, const["NUM_CLASSES"])


def test_MyCNN_step(model, const, helpers):

    fake_batch = helpers.simulate_ecg_signal(const["DURATION"])
    fake_batch["label"] = torch.from_numpy(fake_batch["label"]).unsqueeze(0)
    loss, y_pred, y = model.step(fake_batch, 0)
    assert loss is not None
    assert y_pred.shape == y.shape
