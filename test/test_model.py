import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
#初始化引入路径

from src.model import GoNet
from src.mcts import TreeNode
from constants import *
from utils import *

def test_hello_world_true():
    assert True

def test_create_model():
    model = GoNet(input_channels=1)
    state = create_test_state()
    set_state(state, 7, 7, Chess.WHITE)
    model.eval()
    policy , value = model(state)
    assert policy.shape == (BOARD_SIZE, BOARD_SIZE)
    assert value.shape == (1,)