import torch
import torch.nn as nn
from constants import *
from torch.nn import functional as F

class GoNet(nn.Module):
    def __init__(self, input_channels):
        super(GoNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 策略头
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # 价值头
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 共享特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 策略头
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        policy = policy.view(policy.size(0),BOARD_SIZE,BOARD_SIZE)#输出是一个15*15的张量，每一位表示当前应当在此处落子的概率，范围[0,1]

        # 价值头
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # 输出在[-1,1]之间，表示当前玩家预期胜率，1表示黑棋胜率高

        return policy,value

    def save(self,path='./model_data/model.pt'):
        torch.save(self.state_dict(),path)

    def load(self,path='./model_data/model.pt'):
        self.load_state_dict(torch.load(path))