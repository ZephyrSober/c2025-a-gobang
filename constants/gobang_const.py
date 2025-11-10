from typing import Final
from enum import Enum,IntEnum

BOARD_SIZE: Final = 15
CHESS_NUM: Final = 3
PATTERN: Final = {'live_three':['tccct','tcctct','tctcct'],
                  'live_four':['cccct','tcccc','ccctc','ctccc'],
                  'five':['ccccc']}#t: target;e: empty; c: self chess

class Chess(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    WALL = 3
    def __invert__(self):
        if self == Chess.BLACK:
            return Chess.WHITE
        elif self == Chess.WHITE:
            return Chess.BLACK
        return Chess.EMPTY

    @classmethod
    def from_onehot(cls, onehot):
        if onehot[Chess.BLACK] == 1:
            return Chess.BLACK
        elif onehot[Chess.WHITE] == 1:
            return Chess.WHITE
        return Chess.EMPTY
    @classmethod
    def char_to_chess(cls,char,self_chess):
        if char == 'e':
            return Chess.EMPTY
        elif char == 'c':
            return self_chess
        return Chess.WALL