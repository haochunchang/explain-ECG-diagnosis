import pytest
import sys
from os.path import dirname, isdir, join

from model import MyCNN

sys.path.append(
    dirname(__file__)
)


@pytest.fixture(scope="module")
def const():
    return {
        "NUM_CHANNEL": 15,
        "NUM_CLASSES": 15,
        "DURATION": 2048
    }


@pytest.fixture
def model(const):
    return MyCNN(
        num_channel=const["NUM_CHANNEL"],
        num_class=const["NUM_CLASSES"],
        chunk_size=const["DURATION"]
    )


class Helpers:
    @staticmethod
    def simulate_ecg_signal(duration=2048):
        """
        Generate faked ECG signals by specifying time duration.
        """
        from numpy.random import rand
        from numpy import zeros, float32
        from torch import from_numpy

        fake_signal = rand(1, 15, duration)
        fake_signal = fake_signal.astype(float32)
        fake_label = zeros((15,))
        fake_label[7] = 1
        return {
            "signal": from_numpy(fake_signal),
            "label": fake_label
        }


@pytest.fixture
def helpers():
    return Helpers


@pytest.fixture(scope="function")
def datadir(tmpdir, request):
    '''
    Fixture responsible for moving all contents in test_dir
    to a temporary directory so tests can use them freely.

    Returns:
        tmpdir: py.path.local object, can access temp dir like os.path
    '''
    from distutils import dir_util
    basedir = dirname(request.module.__file__)
    test_dir = basedir
    if isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir
