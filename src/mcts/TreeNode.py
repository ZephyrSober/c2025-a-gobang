import sys
import os
from xmlrpc.client import Fault

import torch
from networkx.algorithms.swap import directed_edge_swap
from sympy import false
from sympy.strategies.core import switch

from utils import drawOnehot, set_state
from math import log

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
#初始化引入路径

from constants import *

class TreeNode:
    def __init__(self, state:torch.Tensor, latest_action:list, parent:'TreeNode', self_chess: Chess):
        self.state = state
        self.valid_range = [[7,7],[7,7]]
        self.parent = parent
        self.latest_action = latest_action
        self.self_chess = self_chess

        self.child_nodes = []
        self.untried_actions = []
        self.is_action_init = False

        self.value = 0
        self.visits = 0
        self.init_valid_range()


    def init_valid_range(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if Chess.from_onehot(self.state[i][j]) == Chess.BLACK or Chess.from_onehot(self.state[i][j]) == Chess.WHITE:
                    self.valid_range[0][0] = i if i < self.valid_range[0][0] else self.valid_range[0][0]
                    self.valid_range[0][1] = j if j < self.valid_range[0][1] else self.valid_range[0][1]
                    self.valid_range[1][0] = i if i > self.valid_range[1][0] else self.valid_range[1][0]
                    self.valid_range[1][1] = j if j > self.valid_range[1][1] else self.valid_range[1][1]

    def init_untried_actions(self) -> None:
        if self.is_action_init:
            return
        self.is_action_init = True

        expand_valid_range = [[self.valid_range[0][0]-EXPLORE_EXPANSION,self.valid_range[1][0]-EXPLORE_EXPANSION],
                                [self.valid_range[0][1]+EXPLORE_EXPANSION+1,self.valid_range[1][1]+EXPLORE_EXPANSION + 1]]

        priority_rules = [(PATTERN["live_four"],True),
                          (PATTERN["live_four"],False),
                          (PATTERN["live_three"],True),
                          (PATTERN["live_three"],False)]
        result = []
        for pattern, by_self_view in priority_rules:
            result = self.find_pos_by_patterns(pattern, by_self_view=by_self_view)
            if len(result) != 0:
                for pos in result:
                    state = self.state.clone()
                    set_state(state, pos[0], pos[1],
                              ~Chess.from_onehot(state[self.latest_action[0], self.latest_action[1]]))
                    self.untried_actions.append(TreeNode(state, pos, parent=self, self_chess=self.self_chess))
                return

        #第五优先级：空位
        for i in range(expand_valid_range[0][0],expand_valid_range[1][0]):
            for j in range(expand_valid_range[0][1], expand_valid_range[1][1]):
                if Chess.from_onehot(self.state[i,j]) == Chess.EMPTY:
                    state = self.state.clone()
                    set_state(state,i,j,~Chess.from_onehot(state[self.latest_action[0],self.latest_action[1]]))
                    # #debug
                    # drawOnehot(state)
                    self.untried_actions.append(TreeNode(state, [i,j],parent= self,self_chess=self.self_chess))

    def find_pos_by_patterns(self,patterns:list,by_self_view:bool) -> list:
        directions = [[1,0],[0,1],[1,1],[-1,1]]
        result = []
        for direction in directions:
            for pattern in patterns:
                for x in range(BOARD_SIZE):
                    for y in range(BOARD_SIZE):
                        if self.is_match_pattern([x,y],direction,pattern,by_self_view=by_self_view):
                            for i,p in enumerate(pattern):
                                if p == 't':
                                    result.append([x+i*direction[0],y+i*direction[1]])
        return result

    def is_match_pattern(self,pos,direction,pattern:str,by_self_view:bool) -> bool:
        x,y = pos
        dx, dy = direction
        pattern = list(pattern)
        for i,p in enumerate(pattern):
            try:
                current_state = Chess.from_onehot(self.state[x+i*dx,y+i*dy])
                if p == 't':
                    p = 'e'
                if Chess.char_to_chess(p, self_chess=self.self_chess if by_self_view else ~self.self_chess) != current_state:
                    return False
            except IndexError:
                return False
        return True



    def get_child_nodes(self) -> list:
        if len(self.child_nodes) != 0:
            return self.child_nodes
        else:
            print("no child node")
            return []

    def pop_one_untried_action(self) ->'TreeNode':
        if not self.is_action_init:
            self.init_untried_actions()
        if self.get_untried_actions_length() != 0:
            return self.untried_actions.pop(0)
        print('no untried action')
        return None

    def get_untried_actions_length(self):
        if not self.is_action_init:
            self.init_untried_actions()
        return len(self.untried_actions)

    def apply_action(self):
        action = self.pop_one_untried_action()
        if action is not None:
            self.child_nodes.append(action)
        return action

    def get_valid_range(self):
        return self.valid_range
    def is_root(self)->bool:
        return True if self.parent is None else False

    def ucb(self,all_time):
        return self.value/self.visits + UCB_C*(log(all_time)/self.visits)**0.5

    def select(self,all_time):
        if self.get_untried_actions_length() != 0:
            return self
        else:
            target = None
            max_ucb = 0
            for node in self.child_nodes:
                if node.ucb(all_time=all_time) >= max_ucb:
                    max_ucb = node.ucb(all_time=all_time)
                    target = node
            return target.select(all_time=all_time)