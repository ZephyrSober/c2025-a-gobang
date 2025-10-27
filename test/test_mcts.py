import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
#初始化引入路径

import torch
from utils import *
from src.mcts import TreeNode
from constants import *

def test_create_tree_node():
    latest_action = [7,7]
    state = create_test_state()
    #create tree node
    node = TreeNode(state,latest_action)

    #test
    assert node.state.shape == (WIDTH,HEIGHT,CHESS_NUM)
    assert node.state[latest_action[0],latest_action[1],Chess.BLACK] == 1

def test_get_child_node():
    latest_action = [7,7]
    state = create_test_state()
    #create tree node
    node = TreeNode(state,latest_action)
    #get children node
    children_node = node.get_child_node()
    first_child_node = children_node[0]
    #debug:
    for child in children_node:
        drawOnehot(child.state)
    #test
    assert len(children_node) == (1+EXPLORE_EXPANSION*2)**2-1
    assert first_child_node.state.shape == (WIDTH,HEIGHT,CHESS_NUM)
    assert first_child_node.state[first_child_node.latest_action[0],first_child_node.latest_action[1],Chess.WHITE] == 1
