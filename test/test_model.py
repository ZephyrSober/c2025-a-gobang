import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
#初始化引入路径

from src.model import gobang

def test_hello_world_true():
    assert True

