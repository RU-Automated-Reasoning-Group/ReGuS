import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import spinup.algos.pytorch.ppo.core as core
from spinup.algos.pytorch.ppo.ppo import ppo
from karel.robot import KarelRobot
from karel.dsl import k_action, k_cond, k_cond_without_not, k_if
from karel.program import AbsState, Branch, Node, Program

robot = KarelRobot(task='topOff', seed=999)

# GT program to solve topOff
program = Program()  # cond == 'front_is_clear'

# main branch
node = Node()
node.action = k_action('move')
abs_state = AbsState()
abs_state.state = {
            'not(front_is_clear)'    : [False],
            'not(left_is_clear)'     : [False],
            'not(right_is_clear)'    : [True],
            'not(no_markers_present)': [False, True],  # NOTE: works for both
}
node.abs_state = abs_state

# sub branch in if
abs_state = AbsState()
abs_state.state = {
            'not(front_is_clear)'    : [False],
            'not(left_is_clear)'     : [False],
            'not(right_is_clear)'    : [True],
            'not(no_markers_present)': [True],
}
branch = Branch(abs_state)
branch.nodes = [Node(abs_state, k_action('put_marker'))]
node.branches = [branch]

program.main = [node]
r, break_point, info = program.execute(robot)


# GT program from scratch
robot = KarelRobot(task='topOff', seed=999)
program = Program()  # cond == 'front_is_clear'

action = k_action('move')
abs_state = AbsState()
abs_state.state = {
            'not(front_is_clear)'    : [False],
            'not(left_is_clear)'     : [False],
            'not(right_is_clear)'    : [True],
            'not(no_markers_present)': [False, True],  # NOTE: works for both
}
program.insert(abs_state=abs_state, action=action, post_abs_state=None)

program.break_point = program.main[0].branches
action = k_action('put_marker')
abs_state = AbsState()
abs_state.state = {
            'not(front_is_clear)'    : [False],
            'not(left_is_clear)'     : [False],
            'not(right_is_clear)'    : [True],
            'not(no_markers_present)': [True],
}
program.insert(abs_state=abs_state, action=action, post_abs_state=None)

r, break_point, info = program.execute(robot)
