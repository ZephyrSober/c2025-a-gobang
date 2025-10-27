import sys
import os

from utils import drawOnehot, set_state

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
#初始化引入路径

from constants import *

class TreeNode:
    def __init__(self, state, latest_action):
        self.state = state
        self.latest_action = latest_action
        self.visits = 0
        self.valid_range = [[7,7],[7,7]]
        for i in range(WIDTH):
            for j in range(HEIGHT):
                if Chess.from_onehot(self.state[i][j]) == Chess.BLACK or Chess.from_onehot(self.state[i][j]) == Chess.WHITE:
                    self.valid_range[0][0] = i if i < self.valid_range[0][0] else self.valid_range[0][0]
                    self.valid_range[0][1] = j if j < self.valid_range[0][1] else self.valid_range[0][1]
                    self.valid_range[1][0] = i if i > self.valid_range[1][0] else self.valid_range[1][0]
                    self.valid_range[1][1] = j if j > self.valid_range[1][1] else self.valid_range[1][1]
        self.child_nodes = []
    def get_child_node(self):
        if len(self.child_nodes) != 0:
            return self.child_nodes
        #第五优先级：空位
        for i in range(self.valid_range[0][0]-EXPLORE_EXPANSION, self.valid_range[1][0]+EXPLORE_EXPANSION + 1):
            for j in range(self.valid_range[0][1]-EXPLORE_EXPANSION, self.valid_range[1][1]+EXPLORE_EXPANSION + 1):
                if Chess.from_onehot(self.state[i,j]) == Chess.EMPTY:
                    state = self.state.clone()
                    set_state(state,i,j,~Chess.from_onehot(state[self.latest_action[0],self.latest_action[1]]))
                    # #debug:
                    # drawOnehot(state)
                    self.child_nodes.append(TreeNode(state, (i,j)))
        return self.child_nodes