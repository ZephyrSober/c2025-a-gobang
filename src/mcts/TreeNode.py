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
        """
        Initializes the untried actions for the current state, considering specific patterns and positions.

        Summary:
        This method initializes a list of untried actions for the current game state. It first checks if the action initialization
        has already been performed. If not, it expands the valid range for exploration and prioritizes certain patterns (like live
        fours and threes) to find potential moves. For each pattern, it generates new states and adds them as untried actions.
        If no high-priority patterns are found, it then considers all empty positions within the expanded range as possible
        untried actions.

        :raises: None

        :returns: None
        """
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
        """
        Finds positions on the board that match given patterns.

        Detailed summary of the function, explaining its purpose and how it operates.
        The function iterates over all possible directions and checks if any of the
        provided patterns match at any position on the board. If a pattern matches,
        it records the positions of 't' (target) in the pattern. The search can be
        conducted from the perspective of the current player or the opponent based
        on the `by_self_view` parameter.

        :param patterns: A list of string patterns to search for on the board.
        :type patterns: list
        :param by_self_view: A boolean indicating whether to search from the
                             perspective of the current player (`True`) or the
                             opponent (`False`).
        :type by_self_view: bool
        :return: A list of [x, y] coordinate pairs where the target 't' in the
                 patterns is found on the board.
        :rtype: list
        """
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
        """
        Checks if the given pattern matches the chess pieces in the specified direction from the starting position.

        :param pos: The starting position as a tuple (x, y).
        :param direction: The direction to check the pattern as a tuple (dx, dy).
        :param pattern: A string representing the pattern of chess pieces.
        :param by_self_view: A boolean indicating whether to use the self view for matching.
        :returns: True if the pattern matches, False otherwise.
        :rtype: bool
        """
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
        """
        Removes and returns one untried action from the list of untried actions.

        :returns: A 'TreeNode' representing the next untried action, or None if no untried actions are left.
        """
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

    def is_terminal(self):
        """
        Checks if the current board state is a terminal state, meaning one of the players
        has won by forming a specific pattern on the board. The function examines all possible
        directions and positions on the board to find a winning pattern for either player.

        :return: A tuple where the first element is the winning player (or None if no winner)
                 and the second element is a boolean indicating if the game is over.
        """
        directions = [[1,0],[0,1],[1,1],[-1,1]]
        pattern = PATTERN["five"][0]
        for direction in directions:
            for x in range(BOARD_SIZE):
                for y in range(BOARD_SIZE):
                    if self.is_match_pattern([x,y],direction,pattern,by_self_view=True):
                        return self.self_chess, True
                    if self.is_match_pattern([x,y],direction,pattern,by_self_view=False):
                        return ~self.self_chess, True
        return None, False


    def ucb(self,all_time):
        return self.value/self.visits + UCB_C*(log(all_time)/self.visits)**0.5

    def select(self,all_time):
        """
        Selects the best child node based on UCB (Upper Confidence Bound) value.

        :param all_time: Total time or iterations for the UCB calculation.
        :type all_time: int
        :return: The selected node with the highest UCB value or self if there are untried actions.
        :rtype: Node

        . note::
           If there are untried actions, the method returns the current node itself.
           Otherwise, it traverses through the child nodes to find the one with the maximum UCB value and continues the selection process from that node.
        """
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