import sys
from distutils import dir_util
from os.path import dirname, isdir, join

import pytest

sys.path.append(
    dirname(join(__file__))
)


@pytest.fixture(scope="function")
def datadir(tmpdir, request):
    '''
    Fixture responsible for moving all contents in test_dir
    to a temporary directory so tests can use them freely.

    Returns:
        tmpdir: py.path.local object, can access temp dir like os.path
    '''
    basedir = dirname(request.module.__file__)
    test_dir = basedir
    if isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir
