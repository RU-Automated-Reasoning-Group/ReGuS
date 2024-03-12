import sys
import random
# import dill as pickle
import argparse
import os
import copy

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MUJOCO_GL'] = 'egl'

import torch
import numpy as np
import gym
from multiworld.envs.mujoco import register_custom_envs as register_mujoco_envs
import matplotlib.pyplot as plt

from ant_program_env import AntProgramEnv
from configs import get_configs
from dsl_ant import *

import pdb

# all primitive policies
ANT_LOW_TORQUE_MODELS = []
for direction in ['up', 'down', 'left', 'right']:
    filename = os.getcwd() + '/primitives/ant_low_torque/' + direction + '.pt'
    ANT_LOW_TORQUE_MODELS.append(torch.load(filename))


def do_ant_test():
    domain = 'AntU'
    num_exps = 2

    # set up the environment
    if domain == 'AntU':
        env = gym.make('AntULongTestEnv-v0', disable_env_checker=True)

    # NOTE: example ground-truth program
    inner_while = WHILE()
    inner_while.cond = [COND_DICT['right_is_clear']]
    inner_while.stmts = [ACTION(ant_action('move'))]
    if_1 = IF()
    if_1.cond = [COND_DICT['right_is_clear']]
    if_1.stmts = [ACTION(ant_action('turn_right')), inner_while]

    inner_while = WHILE()
    inner_while.cond = [COND_DICT['not(front_is_clear)']]
    # inner_while.stmts = [ACTION(ant_action('move'))]
    inner_while.stmts = [C()]
    if_2 = IF()
    if_2.cond = [COND_DICT['not(front_is_clear)']]
    if_2.stmts = [ACTION(ant_action('turn_left'))]

    while_loop = WHILE()
    while_loop.cond = [COND_DICT['not(present_goal)']]
    while_loop.stmts = [
        if_1,
        if_2,
        ACTION(ant_action('move')),
    ]
    
    ant_program = Program()
    ant_program.stmts = [
        while_loop,
        END(),
    ]

    config_dict = get_configs(domain)
    config_dict['max_episode_length'] = 1000
    config_dict['distance_threshold']['front'] = 1.2
    ant_program_env = AntProgramEnv(
        env=env,
        models=ANT_LOW_TORQUE_MODELS,
        goal_threshold=0.5,
        goal_detection_threshold=4.0,
        debug=False,
        **config_dict
        # **get_configs(domain),
    )
    ant_program_env.reset()
    ant_program_env.force_execution = True

    total_reward = 0
    for exp_index in range(num_exps):
        ant_program_env.reset()
        ant_program.execute(ant_program_env)
        reward = ant_program_env.check_reward()
        total_reward += reward

    if total_reward == 2:
        print('Simple Test Ant Case: Success')
    else:
        print('Test Fail')