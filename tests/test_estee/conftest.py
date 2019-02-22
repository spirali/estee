import os
import sys

TEST_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(TEST_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from .test_utils import plan1  # noqa
