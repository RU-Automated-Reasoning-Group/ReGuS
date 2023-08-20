import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F

from karel.dsl import *


# actions
# 0 : move
# 1 : turn_right
# 2 : turn_left
# 3 : pick_marker
# 4 : put_marker

ACTION_INDEX = [0, 1, 2, 3, 4]
ACTION_NAME = [
    'move',
    'turn_right',
    'turn_left',
    'pick_marker',
    'put_marker'
]
CODE_DIM = 5


class OldEnvEncoder(nn.Module):
    def __init__(self, h, w):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                        kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * h * w, 512),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        avg_x = torch.zeros(1, x.shape[1])
        for i in range(batch_size):
            avg_x += x[i, :]

        return avg_x / batch_size


class EnvEncoder(nn.Module):
    def __init__(self, h, w):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                        kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                        kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 512),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        avg_x = torch.zeros(1, x.shape[1])
        for i in range(batch_size):
            avg_x += x[i, :]

        return avg_x / batch_size


class ValueFunction(nn.Module):
    def __init__(self):
        super().__init__()

        self.w1 = nn.Linear(512, 256)
        self.w2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.w1(x)
        x = self.relu(x)
        x = self.w2(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class CodePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = MLP(input_dim=512, output_dim=CODE_DIM)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, eps=0.05):

        logits = self.mlp(x)
        ll = self.log_softmax(logits)
        scores = torch.exp(ll) * (1 - eps) + eps / ll.shape[1]
    
        c = Categorical(scores)
        index = c.sample()
        nll = F.nll_loss(ll, index)

        return index, nll
