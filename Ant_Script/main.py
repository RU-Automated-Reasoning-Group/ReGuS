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
from dsl import *

# all primitive policies
ANT_LOW_TORQUE_MODELS = []
for direction in ['up', 'down', 'left', 'right']:
    filename = os.getcwd() + '/primitives/ant_low_torque/' + direction + '.pt'
    ANT_LOW_TORQUE_MODELS.append(torch.load(filename))


def main(args):

    domain = args.domain
    num_exps = args.num_exps

    # set up the environment
    if domain == 'AntU':
        env = gym.make('AntULongTestEnv-v0', disable_env_checker=True)

    elif domain == 'AntFb':
        env = gym.make('AntFbMedTestEnv-v1')
        
    elif domain == 'AntMaze':
        env = gym.make('AntMazeMedTestEnv-v1')
        
    elif domain == 'AntFg':
        env = gym.make('AntFgMedTestEnv-v1')
     
    # NOTE: example ground-truth program
    move = ACTION(ant_action('move'))
    turn_left = ACTION(ant_action('turn_left'))
    turn_right = ACTION(ant_action('turn_right'))

    not_front_is_clear = COND_DICT['not(front_is_clear)']
    right_is_clear = COND_DICT['right_is_clear']
    left_is_clear = COND_DICT['left_is_clear']
    not_present_goal = COND_DICT['not(present_goal)']

    # NOTE: test this program
    # WHILE(not (present_goal)) {
    #   IF(right_is_clear) { 
    #      turn_right 
    #      WHILE(right_is_clear) {move} 
    #   }
    #   IF(not front_is_clear) {
    #       turn_left
    #   }
    #   move 
    # }

    inner_while = WHILE()
    inner_while.cond = [copy.deepcopy(right_is_clear)]
    inner_while.stmts = [move]

    if_1 = IF()
    if_1.cond = [copy.deepcopy(right_is_clear)]
    if_1.stmts = [turn_right, inner_while]

    if_2 = IF()
    if_2.cond = [copy.deepcopy(not_front_is_clear)]
    if_2.stmts = [turn_left]

    while_loop = WHILE()
    while_loop.cond = [copy.deepcopy(not_present_goal)]
    while_loop.stmts = [
        if_1,
        if_2,
        move,
    ]
    
    ant_program = Program()
    ant_program.stmts = [
        while_loop,
        S(),
        END(),
    ]

    ant_program_env = AntProgramEnv(
        env=env,
        models=ANT_LOW_TORQUE_MODELS,
        goal_threshold=0.5,
        goal_detection_threshold=4.0,
        **get_configs(domain),
    )
    ant_program_env.reset()
    ant_program_env.force_execution = True

    total_reward = 0
    for exp_index in range(num_exps):
        print('[TRAJECTORY][{}]'.format(exp_index))
        ant_program_env.reset()
        ant_program.execute(ant_program_env)
        reward = ant_program_env.check_reward()
        total_reward += reward

    print('{} success rate: {} / {} = {:.2%}'.format(domain, total_reward, num_exps, total_reward/num_exps))
    

def main_2(args):

    domain = args.domain
    num_exps = args.num_exps

    # set up the environment
    if domain == 'AntU':
        env = gym.make('AntULongTestEnv-v0', disable_env_checker=True)

    elif domain == 'AntFb':
        env = gym.make('AntFbMedTestEnv-v1')
        
    elif domain == 'AntMaze':
        env = gym.make('AntMazeMedTestEnv-v1')
        
    elif domain == 'AntFg':
        env = gym.make('AntFgMedTestEnv-v1')
     
    # NOTE: example ground-truth program
    move = ACTION(ant_action('move'))
    turn_left = ACTION(ant_action('turn_left'))
    turn_right = ACTION(ant_action('turn_right'))

    not_front_is_clear = COND_DICT['not(front_is_clear)']
    right_is_clear = COND_DICT['right_is_clear']
    left_is_clear = COND_DICT['left_is_clear']
    not_present_goal = COND_DICT['not(present_goal)']

    # NOTE: test this program
    # WHILE(not (present_goal)) {
    #   IF(right_is_clear) { 
    #      turn_right 
    #      WHILE(right_is_clear) {move} 
    #   }
    #   IF(not front_is_clear) {
    #       turn_left
    #   }
    #   move 
    # }

    inner_while = WHILE()
    inner_while.cond = [COND_DICT['not(front_is_clear)']]
    inner_while.stmts = [Stop(), move]

    if_1 = IF()
    if_1.cond = [COND_DICT['not(front_is_clear)']]
    if_1.stmts = [turn_left, inner_while]

    while_loop = WHILE()
    while_loop.cond = [COND_DICT['not(present_goal)']]
    while_loop.stmts = [
        if_1,
        move,
    ]
    
    ant_program = Program()
    ant_program.stmts = [
        while_loop,
        END(),
    ]

    print(ant_program)
    pdb.set_trace()

    ant_program_env = AntProgramEnv(
        env=env,
        models=ANT_LOW_TORQUE_MODELS,
        goal_threshold=0.5,
        goal_detection_threshold=4.0,
        **get_configs(domain),
    )
    ant_program_env.reset()
    ant_program_env.force_execution = True

    total_reward = 0
    for exp_index in range(num_exps):
        print('[TRAJECTORY][{}]'.format(exp_index))
        ant_program_env.reset()
        ant_program.execute(ant_program_env)
        # ant_program_env.plot_trajectory(domain=domain, exp_index=exp_index)
        reward = ant_program_env.check_reward()
        total_reward += reward

    print('{} success rate: {} / {} = {:.2%}'.format(domain, total_reward, num_exps, total_reward/num_exps))


def main_3(args):

    domain = args.domain
    num_exps = args.num_exps

    # set up the environment
    if domain == 'AntU':
        env = gym.make('AntULongTestEnv-v0', disable_env_checker=True)

    elif domain == 'AntFb':
        env = gym.make('AntFbMedTestEnv-v1', disable_env_checker=True)
        
    elif domain == 'AntMaze':
        env = gym.make('AntMazeMedTestEnv-v1', disable_env_checker=True)
        
    elif domain == 'AntFg':
        env = gym.make('AntFgMedTestEnv-v1', disable_env_checker=True)
     
    # NOTE: example ground-truth program
    move = ACTION(ant_action('move'))
    turn_left = ACTION(ant_action('turn_left'))
    turn_right = ACTION(ant_action('turn_right'))

    not_front_is_clear = COND_DICT['not(front_is_clear)']
    right_is_clear = COND_DICT['right_is_clear']
    left_is_clear = COND_DICT['left_is_clear']
    not_present_goal = COND_DICT['not(present_goal)']

    # NOTE: test this program
    # WHILE(not (present_goal)) {
    #   IF(right_is_clear) { 
    #      turn_right 
    #      WHILE(right_is_clear) {move} 
    #   }
    #   IF(not front_is_clear) {
    #       turn_left
    #   }
    #   move 
    # }

    inner_while = WHILE()
    inner_while.cond = [COND_DICT['not(front_is_clear)']]
    inner_while.stmts = [move]

    if_1 = IF()
    if_1.cond = [COND_DICT['not(right_is_clear)']]
    if_1.stmts = [turn_left, inner_while]

    if_out = IF()
    if_out.cond = [COND_DICT['not(front_is_clear)']]
    if_out.stmts = [if_1]

    inner_while_2 = WHILE()
    inner_while_2.cond = [COND_DICT['not(front_is_clear)']]
    inner_while_2.stmts = [C()]
    if_2 = IF()
    if_2.cond = [COND_DICT['not(front_is_clear)']]
    if_2.stmts = [turn_right, inner_while_2]

    while_loop = WHILE()
    while_loop.cond = [COND_DICT['not(present_goal)']]
    while_loop.stmts = [
        if_out,
        if_2,
        move,
    ]
    
    ant_program = Program()
    ant_program.stmts = [
        while_loop,
        END(),
    ]

    # WHILE(not (present_goal)) { 
    #     IF(not (front_is_clear)) { 
    #         IF(not (right_is_clear)) { 
    #             turn_left 
    #             WHILE(not (front_is_clear)) { move} ;
    #         } 
    #     }  
    #     IF(not (front_is_clear)) { 
    #         turn_right 
    #         WHILE(not (front_is_clear)) { C } ;
    #     }  
    #     move
    # } ; C ; END

    print(ant_program)
    pdb.set_trace()

    ant_program_env = AntProgramEnv(
        env=env,
        models=ANT_LOW_TORQUE_MODELS,
        goal_threshold=0.5,
        goal_detection_threshold=4.0,
        **get_configs(domain),
    )
    ant_program_env.reset()
    ant_program_env.force_execution = True

    total_reward = 0
    for exp_index in range(num_exps):
        print('[TRAJECTORY][{}]'.format(exp_index))
        ant_program_env.reset()
        ant_program.execute(ant_program_env)
        # ant_program_env.plot_trajectory(domain=domain, exp_index=exp_index)
        reward = ant_program_env.check_reward()
        total_reward += reward
        if reward != 1:
            ant_program_env.plot_trajectory(domain=domain, exp_index=exp_index)
            pdb.set_trace()

    print('{} success rate: {} / {} = {:.2%}'.format(domain, total_reward, num_exps, total_reward/num_exps))


# WHILE(not (present_goal)) { 
#     IF(not (front_is_clear)) { 
#         IF(left_is_clear) { 
#             turn_left 
#             WHILE(not (front_is_clear)) { C } ;
#         } 
#     }  
#     IF(not (front_is_clear)) { 
#         turn_right 
#         WHILE(not (front_is_clear)) { move} 
#     ;}  
#     move
# } ; C ; END
def main_4(args):

    domain = args.domain
    num_exps = args.num_exps

    # set up the environment
    if domain == 'AntU':
        env = gym.make('AntULongTestEnv-v0', disable_env_checker=True)

    elif domain == 'AntFb':
        env = gym.make('AntFbMedTestEnv-v1', disable_env_checker=True)
        
    elif domain == 'AntMaze':
        env = gym.make('AntMazeMedTestEnv-v1', disable_env_checker=True)
        
    elif domain == 'AntFg':
        env = gym.make('AntFgMedTestEnv-v1', disable_env_checker=True)
     
    # NOTE: example ground-truth program
    inner_while = WHILE()
    inner_while.cond = [COND_DICT['not(front_is_clear)']]
    inner_while.stmts = [C()]

    if_1 = IF()
    if_1.cond = [COND_DICT['left_is_clear']]
    if_1.stmts = [ACTION(ant_action('turn_left')), inner_while]

    if_out = IF()
    if_out.cond = [COND_DICT['not(front_is_clear)']]
    if_out.stmts = [if_1]

    inner_while_2 = WHILE()
    inner_while_2.cond = [COND_DICT['not(front_is_clear)']]
    inner_while_2.stmts = [ACTION(ant_action('move'))]
    if_2 = IF()
    if_2.cond = [COND_DICT['not(front_is_clear)']]
    if_2.stmts = [ACTION(ant_action('turn_right')), inner_while_2]

    while_loop = WHILE()
    while_loop.cond = [COND_DICT['not(present_goal)']]
    while_loop.stmts = [
        if_out,
        if_2,
        ACTION(ant_action('move')),
    ]
    
    ant_program = Program()
    ant_program.stmts = [
        while_loop,
        END(),
    ]

    print(ant_program)
    pdb.set_trace()

    ant_program_env = AntProgramEnv(
        env=env,
        models=ANT_LOW_TORQUE_MODELS,
        goal_threshold=0.5,
        goal_detection_threshold=4.0,
        **get_configs(domain),
    )
    ant_program_env.reset()
    ant_program_env.force_execution = True

    total_reward = 0
    for exp_index in range(num_exps):
        print('[TRAJECTORY][{}]'.format(exp_index))
        ant_program_env.reset()
        ant_program.execute(ant_program_env)
        # ant_program_env.plot_trajectory(domain=domain, exp_index=exp_index)
        reward = ant_program_env.check_reward()
        total_reward += reward
        if reward != 1:
            ant_program_env.plot_trajectory(domain=domain, exp_index=exp_index)

    print('{} success rate: {} / {} = {:.2%}'.format(domain, total_reward, num_exps, total_reward/num_exps))


# WHILE(not (present_goal)) { 
#     IF(right_is_clear) { 
#         turn_right 
#         WHILE(right_is_clear) { move} 
#     ;}  
#     IF(not (front_is_clear)) { 
#         turn_left 
#         WHILE(not (front_is_clear)) { move} 
#     ;}  
#     move
# } ; C ; END
def main_5(args):
    domain = args.domain
    num_exps = args.num_exps

    # set up the environment
    if domain == 'AntU':
        env = gym.make('AntULongTestEnv-v0', disable_env_checker=True)

    elif domain == 'AntFb':
        env = gym.make('AntFbMedTestEnv-v1', disable_env_checker=True)
        
    elif domain == 'AntMaze':
        env = gym.make('AntMazeMedTestEnv-v1', disable_env_checker=True)
        
    elif domain == 'AntFg':
        env = gym.make('AntFgMedTestEnv-v1', disable_env_checker=True)
     
    # NOTE: example ground-truth program
    inner_while = WHILE()
    inner_while.cond = [COND_DICT['right_is_clear']]
    inner_while.stmts = [ACTION(ant_action('move'))]

    if_1 = IF()
    if_1.cond = [COND_DICT['right_is_clear']]
    if_1.stmts = [ACTION(ant_action('turn_right')), inner_while]

    inner_while_2 = WHILE()
    inner_while_2.cond = [COND_DICT['not(front_is_clear)']]
    inner_while_2.stmts = [ACTION(ant_action('move'))]
    if_2 = IF()
    if_2.cond = [COND_DICT['not(front_is_clear)']]
    # if_2.stmts = [ACTION(ant_action('turn_left')), inner_while_2]
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

    print(ant_program)
    pdb.set_trace()

    ant_program_env = AntProgramEnv(
        env=env,
        models=ANT_LOW_TORQUE_MODELS,
        goal_threshold=0.5,
        goal_detection_threshold=4.0,
        **get_configs(domain),
    )
    ant_program_env.reset()
    ant_program_env.force_execution = True

    total_reward = 0
    for exp_index in range(num_exps):
        print('[TRAJECTORY][{}]'.format(exp_index))
        ant_program_env.reset()
        ant_program.execute(ant_program_env)
        # ant_program_env.plot_trajectory(domain=domain, exp_index=exp_index)
        reward = ant_program_env.check_reward()
        total_reward += reward
        # if reward != 1:
        #     ant_program_env.plot_trajectory(domain=domain, exp_index=exp_index)

    print('{} success rate: {} / {} = {:.2%}'.format(domain, total_reward, num_exps, total_reward/num_exps))

# WHILE(not (present_goal)) { 
#   IF(not (front_is_clear)) { 
#       IF(not (right_is_clear)) { 
#           turn_left 
#           WHILE(not (front_is_clear)) { move} ;
#       } 
#   }  
#   IF(not (front_is_clear)) { 
#       turn_right 
#       WHILE(not (front_is_clear)) { C } ;
#   }  
#   move
# } ; C ; END

# WHILE(not (present_goal)) { IF(not (front_is_clear)) { IF(not (left_is_clear)) { move WHILE(not (left_is_clear)) { IF(front_is_clear) { WHILE(front_is_clear) { move} ; turn_right}  move} ; move turn_right}  IF(not (front_is_clear)) { turn_left WHILE(not (front_is_clear)) { S } ;} }  move} ;; END
# WHILE(not (present_goal)) { IF(not (front_is_clear)) { IF(not (right_is_clear)) { turn_left WHILE(not (front_is_clear)) { move} ;} }  IF(not (front_is_clear)) { turn_right WHILE(not (front_is_clear)) { C } ;}  move} ; C ; END

def main_6(args):
    domain = args.domain
    num_exps = args.num_exps

    # set up the environment
    if domain == 'AntU':
        env = gym.make('AntULongTestEnv-v0', disable_env_checker=True)

    elif domain == 'AntFb':
        env = gym.make('AntFbMedTestEnv-v1', disable_env_checker=True)
        
    elif domain == 'AntMaze':
        env = gym.make('AntMazeMedTestEnv-v1', disable_env_checker=True)
        
    elif domain == 'AntFg':
        env = gym.make('AntFgMedTestEnv-v1', disable_env_checker=True)
     
    # NOTE: example ground-truth program
    inner_while = WHILE()
    inner_while.cond = [COND_DICT['not(right_is_clear)']]
    inner_while.stmts = [ACTION(ant_action('move'))]
    if_1 = IF()
    if_1.cond = [COND_DICT['not(right_is_clear)']]
    if_1.stmts = [ACTION(ant_action('turn_left')), inner_while]
    if_2 = IF()
    if_2.cond = [COND_DICT['not(front_is_clear)']]
    if_2.stmts = [if_1]

    inner_while_2 = WHILE()
    inner_while_2.cond = [COND_DICT['not(front_is_clear)']]
    if_3 = IF()
    if_3.cond = [COND_DICT['not(front_is_clear)']]
    if_3.stmts = [ACTION(ant_action('turn_right')), inner_while_2]

    while_loop = WHILE()
    while_loop.cond = [COND_DICT['not(present_goal)']]
    while_loop.stmts = [
        if_2,
        if_3,
        ACTION(ant_action('move')),
    ]
    
    ant_program = Program()
    ant_program.stmts = [
        while_loop,
        END(),
    ]

    print(ant_program)
    pdb.set_trace()

    ant_program_env = AntProgramEnv(
        env=env,
        models=ANT_LOW_TORQUE_MODELS,
        goal_threshold=0.5,
        goal_detection_threshold=4.0,
        **get_configs(domain),
    )
    ant_program_env.reset()
    ant_program_env.force_execution = True

    total_reward = 0
    for exp_index in range(num_exps):
        print('[TRAJECTORY][{}]'.format(exp_index))
        ant_program_env.reset()
        ant_program.execute(ant_program_env)
        # ant_program_env.plot_trajectory(domain=domain, exp_index=exp_index)
        reward = ant_program_env.check_reward()
        total_reward += reward
        # if reward != 1:
        #     ant_program_env.plot_trajectory(domain=domain, exp_index=exp_index)

    print('{} success rate: {} / {} = {:.2%}'.format(domain, total_reward, num_exps, total_reward/num_exps))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, choices=('AntU', 'AntFb', 'AntMaze', 'AntFg'), default='AntU')
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--num_exps', type=int, default=5)
    parser.add_argument('--save_trajectory', dest='save_trajectory', action='store_true')
    parser.add_argument('--save_trajectory_gif', dest='save_trajectory_gif', action='store_true')
    args = parser.parse_args()
    
    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # main(args)
    # main_2(args)
    # main_3(args)
    # main_4(args)
    # main_5(args)
    main_6(args)