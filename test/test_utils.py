import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
#初始化引入路径

import torch
from constants import *
from utils import drawOnehot, create_test_state


def test_drawOnehot():
    state = create_test_state()

    assert drawOnehot(state,force_plot= False)