import sys
import os
from operator import truediv

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
    set_state(state, 7, 7, Chess.BLACK)
    #create tree node
    node = TreeNode(state,latest_action, None,self_chess=Chess.BLACK)

    #test
    assert node.state.shape == (BOARD_SIZE,BOARD_SIZE,CHESS_NUM)
    assert node.state[latest_action[0],latest_action[1],Chess.BLACK] == 1

def test_get_child_node():
    latest_action = [7,7]
    state = create_test_state()
    set_state(state, 7, 7, Chess.BLACK)
    #create tree node
    node = TreeNode(state,latest_action, None,self_chess=Chess.BLACK)
    #set child node
    node.apply_action()
    #get children node
    children_node = node.get_child_nodes()
    first_child_node = children_node[0]
    # #debug:
    # for child in children_node:
    #     drawOnehot(child.state)
    #test
    assert len(node.untried_actions) == (1+EXPLORE_EXPANSION*2)**2-1-1
    assert first_child_node.state.shape == (BOARD_SIZE,BOARD_SIZE,CHESS_NUM)
    assert first_child_node.state[first_child_node.latest_action[0],first_child_node.latest_action[1],Chess.WHITE] == 1
    assert first_child_node.parent is node

def test_select_single_root():
    state = create_test_state()
    set_state(state, 7, 7, Chess.BLACK)
    latest_action= [7,7]
    root = TreeNode(state,latest_action,None,self_chess=Chess.BLACK)
    root.apply_action()
    selected = root.select(all_time=2)
    assert selected is root

def test_select_by_ucb():
    state = create_test_state()
    set_state(state, 7, 7, Chess.BLACK)
    latest_action = [7,7]
    root = TreeNode(state,latest_action,None,self_chess=Chess.BLACK)
    root.visits = 1
    root.value = 1
    child1 = root.apply_action()
    child1.visits = 1
    child1.value = 1
    print("child1 is", child1)
    child2 = root.apply_action()
    child2.visits = 2
    child2.value = 1
    print("child2 is", child2)
    print("root is", root)
    for i in range(len(root.untried_actions)):
        root.apply_action().visits = 1
    assert root.select(all_time=9) is child1

def test_match_pattern_by_self_view():
    state = create_test_state()
    set_state(state, 7, 7, Chess.BLACK)
    set_state(state,6, 6, Chess.BLACK)
    set_state(state,5, 5, Chess.BLACK)
    set_state(state,4, 4, Chess.BLACK)
    set_state(state,3, 3, Chess.WHITE)
    latest_action = [7,7]
    node = TreeNode(state,latest_action,None,self_chess=Chess.BLACK)
    assert node.is_match_pattern([4,4], [1,1], PATTERN['live_four'][0],True) == True
    assert node.is_match_pattern([3,3], [1,1], PATTERN['live_three'][0],True) == False

def test_match_pattern_by_opponent_view():
    state = create_test_state()
    set_state(state, 7, 7, Chess.BLACK)
    set_state(state,6, 6, Chess.BLACK)
    set_state(state,5, 5, Chess.BLACK)
    set_state(state,4, 4, Chess.BLACK)
    set_state(state,3, 3, Chess.WHITE)
    latest_action = [7,7]
    node = TreeNode(state,latest_action,None,self_chess=Chess.WHITE)
    assert node.is_match_pattern([4,4], [1,1], PATTERN['live_four'][0],True) == False
    assert node.is_match_pattern([4,4], [1,1], PATTERN['live_four'][0],False) == True

def test_find_pos_by_patterns():
    state = create_test_state()
    set_state(state, 7, 7, Chess.BLACK)
    set_state(state, 6, 6, Chess.BLACK)
    set_state(state, 5, 5, Chess.BLACK)
    set_state(state, 4, 4, Chess.BLACK)
    set_state(state, 3, 3, Chess.WHITE)
    set_state(state, 4, 5, Chess.BLACK)
    set_state(state, 4, 6, Chess.BLACK)
    latest_action = [7, 7]
    node = TreeNode(state, latest_action, None, self_chess=Chess.WHITE)
    assert node.find_pos_by_patterns(PATTERN['live_four'],False) == [[8,8]]
    assert node.find_pos_by_patterns(PATTERN['live_three'],False) == [[4,3],[4,7]]



def test_get_action_black_live_four():
    state = create_test_state()
    set_state(state, 7, 7, Chess.BLACK)
    set_state(state, 6, 6, Chess.BLACK)
    set_state(state, 5, 5, Chess.BLACK)
    set_state(state, 4, 4, Chess.BLACK)
    set_state(state, 3, 3, Chess.WHITE)
    latest_action = [7,7]
    node = TreeNode(state,latest_action,None, Chess.BLACK)
    assert node.pop_one_untried_action().latest_action == [8,8]

def test_is_terminal():
    state = create_test_state()
    set_state(state, 7, 7, Chess.BLACK)
    set_state(state, 6, 6, Chess.BLACK)
    set_state(state, 5, 5, Chess.BLACK)
    set_state(state, 4, 4, Chess.BLACK)
    latest_action = [7,7]
    node = TreeNode(state, latest_action, None, Chess.BLACK)
    winner = node.get_terminal_player()
    assert winner is None
    set_state(state, 3, 3, Chess.BLACK)
    winner = node.get_terminal_player()
    assert winner is Chess.BLACK

def test_stimulate():
    state = create_test_state()
    set_state(state, 7, 7, Chess.BLACK)
    latest_action = [7,7]
    node = TreeNode(state,latest_action,None,self_chess=Chess.BLACK)
    winner , final_state= node.stimulate()
    print(winner)
    print(final_state.latest_action)
    assert drawOnehot(final_state.state)

def test_back_propagate():
    state = create_test_state()
    set_state(state, 7, 7, Chess.BLACK)
    latest_action = [7,7]
    node = TreeNode(state,latest_action,None,self_chess=Chess.BLACK)
    winner, final_state = node.stimulate()
    final_state.back_propagate()
    assert node.visits == 1
#todo: a tool set to monitor the tree flow