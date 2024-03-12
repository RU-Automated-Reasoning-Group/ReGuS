import time
import copy
import random
from tqdm import tqdm

import numpy as np

from dsl_karel_new import *
from karel.robot import KarelRobot
from utils.logging import init_logging, log_and_print

import pdb

def test_large_maze():
    if_1 = IF(COND_DICT['right_is_clear'])
    if_1.stmts = [ACTION(ACTION_DICT['turn_right'])]
    if_2 = IF(COND_DICT['not(front_is_clear)'])
    if_2.stmts = [ACTION(ACTION_DICT['turn_left'])]
    a_1 = ACTION(ACTION_DICT['move'])
    while_1 = WHILE()
    while_1.cond = [COND_DICT['not(markers_present)']]
    while_1.stmts = [if_1, if_2, a_1]

    program = Program()
    program.stmts = [while_1, program.stmts[-1]]

    rewards = []
    eval_seeds = [10000 + 1000 * i for i in range(1)]
    for seed in tqdm(eval_seeds):
        force_eval_robot = KarelRobot(task='randomMaze', seed=seed)
        force_eval_robot.max_steps = 100000
        force_eval_robot.force_execution = True
        program.execute(force_eval_robot)
        program.reset()
        r = force_eval_robot.check_reward()

    if r == 1:
        print('Simple Test Karel Case: Success')
    else:
        print('Test Fail')