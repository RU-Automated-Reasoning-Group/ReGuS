import copy

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.dropout import Dropout

from karel.dsl import *


# reusable code snippet for deriving k_place_holder()
CODE = []

p_actions = [
    k_action('move'),
    k_action('turn_right'),
    k_action('turn_left'),
    k_action('pick_marker'),
    k_action('put_marker')
]

p_conds = [
    # cond
    k_cond(negation=False, cond=k_cond_without_not('front_is_clear')),
    k_cond(negation=False, cond=k_cond_without_not('left_is_clear')),
    k_cond(negation=False, cond=k_cond_without_not('right_is_clear')),
    k_cond(negation=False, cond=k_cond_without_not('markers_present')),
    k_cond(negation=False, cond=k_cond_without_not('no_markers_present')),
    # NOT cond
    k_cond(negation=True, cond=k_cond_without_not('front_is_clear')),
    k_cond(negation=True, cond=k_cond_without_not('left_is_clear')),
    k_cond(negation=True, cond=k_cond_without_not('right_is_clear')),
    k_cond(negation=True, cond=k_cond_without_not('markers_present')),
    k_cond(negation=True, cond=k_cond_without_not('no_markers_present')),
]


def p_ph_stmt_stmt():
    p = k_stmt_stmt(
        k_stmt(
            k_place_holder()
        ),
        k_stmt(
            k_place_holder()
        )
    )

    return copy.deepcopy(p)


def p_ph_while(cond_index):
    p = k_while(
        copy.deepcopy(p_conds[cond_index]),  # choose one from p_conds
        k_stmt(
            k_place_holder()
        )
    )

    return copy.deepcopy(p)


def p_ph_if(cond_index):
    p = k_if(
        copy.deepcopy(p_conds[cond_index]),  # choose one from p_conds
        k_stmt(
            k_place_holder()
        )
    )

    return copy.deepcopy(p)


def p_ph_ifelse(cond_index):
    p = k_ifelse(
        copy.deepcopy(p_conds[cond_index]),  # choose one from p_conds
        k_stmt(
            k_place_holder()
        ),
        k_stmt(
            k_place_holder()
        )
    )

    return p


def p_ph_action(action_index):
    p = copy.deepcopy(p_actions[action_index])

    return p


# NO stmt_stmt

# 0 : move
# 1 : turn_right
# 2 : turn_left
# 3 : pick_marker
# 4 : put_marker

# 5 : stmt { while (cond) {stmt{PH}} }
# ...
# 14

# 15 : stmt { if (cond) {stmt{PH}} }
# ...
# 24

# 25 : stmt { ifelse (cond) {stmt{PH}} else {stmt{PH}}}
# ...
# 34 :


# stmt_stmt
# 35 ~ 69
# stmt_stmt { {0~34}, stmt{PH} }

for p in p_actions:
    CODE.append(p)

for cond_index in range(len(p_conds)):
    CODE.append(p_ph_while(cond_index))

for cond_index in range(len(p_conds)):
    CODE.append(p_ph_if(cond_index))

for cond_index in range(len(p_conds)):
    CODE.append(p_ph_ifelse(cond_index))

for p in p_actions:
    ss = p_ph_stmt_stmt()
    ss.stmt1 = p
    CODE.append(ss)

for cond_index in range(len(p_conds)):
    ss = p_ph_stmt_stmt()
    ss.stmt1 = p_ph_while(cond_index)
    CODE.append(ss)

for cond_index in range(len(p_conds)):
    ss = p_ph_stmt_stmt()
    ss.stmt1 = p_ph_if(cond_index)
    CODE.append(ss)

for cond_index in range(len(p_conds)):
    ss = p_ph_stmt_stmt()
    ss.stmt1 = p_ph_ifelse(cond_index)
    CODE.append(ss)


class EnvEncoder(nn.Module):
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
        x = self.model(x)
        avg_x = torch.zeros(1, x.shape[1])
        for i in range(batch_size):
            avg_x += x[i, :]

        return avg_x / batch_size


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class CodePredictor(nn.Module):
    def __init__(self, h, w):
        super().__init__()

        self.env_encoder = EnvEncoder(h=h, w=w)
        self.mlp = MLP(input_dim=512, output_dim=70)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        
        x = self.env_encoder(x)
        x = self.mlp(x)
        x = self.softmax(x)
        
        c = Categorical(x)
        index = c.sample()
        if mask:
            while index == mask:
                index = c.sample()

        return copy.deepcopy(CODE[index]), index