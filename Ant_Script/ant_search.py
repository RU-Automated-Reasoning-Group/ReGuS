import os
import gym
import time
import copy
import torch
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.logging import init_logging, log_and_print
from utils.parse_lib import get_parse
from dsl import *
from search_karel_new_while import Node
from configs import get_configs
from ant_program_env import AntProgramEnv

import pdb

# all primitive policies
ANT_LOW_TORQUE_MODELS = []
for direction in ['up', 'down', 'left', 'right']:
    filename = os.getcwd() + '/primitives/ant_low_torque/' + direction + '.pt'
    ANT_LOW_TORQUE_MODELS.append(torch.load(filename))

def var_plot(all_time_steps, all_rewards, fig_path):
    max_time_steps = None
    entire_time_steps = []

    # get max time steps
    for time_steps in all_time_steps:
        if max_time_steps is None:
            max_time_steps = time_steps[-1] + 1e4
        else:
            max_time_steps = max(max_time_steps, time_steps[-1]) + 1e4
        entire_time_steps += time_steps + [max_time_steps]

    # clean reward
    entire_time_steps = np.array(sorted(list(set(entire_time_steps))))
    clean_all_rewards = []
    for time_step, reward in zip(all_time_steps, all_rewards):
        cur_time_step_id = 0
        new_reward = []
        for need_step in entire_time_steps:
            # next
            if cur_time_step_id < len(time_step) and need_step > time_step[cur_time_step_id]:
                cur_time_step_id += 1
            while cur_time_step_id < len(time_step) and need_step > time_step[cur_time_step_id]:
                assert time_step[cur_time_step_id-1] == time_step[cur_time_step_id]
                cur_time_step_id += 1
            # store
            if cur_time_step_id >= len(time_step):
                new_reward.append(reward[-1])
            else:
                new_reward.append(reward[cur_time_step_id])
        clean_all_rewards.append(new_reward)

    # plot
    entire_time_steps = np.array(entire_time_steps)
    clean_all_rewards = np.array(clean_all_rewards)

    plt.figure()
    plt.plot(entire_time_steps, np.mean(clean_all_rewards, axis=0), 'r-', label='ReGuS')
    plt.fill_between(entire_time_steps, \
                     np.min(clean_all_rewards, axis=0), \
                     np.max(clean_all_rewards, axis=0), \
                     alpha=0.2, facecolor='r')

    plt.legend(fontsize=14)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.savefig(fig_path)


def gt_structure():
    while_1 = WHILE()
    while_1.cond = [COND_DICT['not(present_goal)']]
    while_1.stmts = [C()]

    program = Program()
    program.stmts = [while_1, C(), program.stmts[-1]]

    return program

# helper function for clean while
def clean_while(stmts):
    drop_ids = []
    for idx, code in enumerate(stmts):
        if isinstance(code, (C, S)):
            drop_ids.append(idx)
        elif isinstance(code, WHILE):
            if len(code.stmts) == 1 and isinstance(code.stmts[0], (C, S)):
                drop_ids.append(idx)
            else:
                clean_while(code.stmts)
        elif isinstance(code, IF):
            clean_while(code.stmts)
        elif isinstance(code, IFELSE):
            clean_while(code.stmts)
            clean_while(code.else_stmts)
        
    # do drop
    for drop_id in drop_ids[::-1]:
        stmts.pop(drop_id)

def eval_program(ant_program):
    # clean program first
    clean_while(ant_program.stmts)

    task_solve = 0
    tasks = ['AntU', 'AntFb', 'AntFg', 'AntMaze']
    for task in tasks:
        # create gym environment
        if task == 'AntU':
            env = gym.make('AntULongTestEnv-v0', disable_env_checker=True)
        elif task == 'AntFb':
            env = gym.make('AntFbMedTestEnv-v1', disable_env_checker=True)
        elif task == 'AntMaze':
            env = gym.make('AntMazeMedTestEnv-v1', disable_env_checker=True)
        elif task == 'AntFg':
            env = gym.make('AntFgMedTestEnv-v1', disable_env_checker=True)

        # evaluate environment
        config_dict = get_configs(task)
        config_dict['distance_threshold']['front'] = 1.2
        if task == 'AntMaze':
            config_dict['max_episode_length'] = 1000
        ant_program_env = AntProgramEnv(
            env=env,
            models=ANT_LOW_TORQUE_MODELS,
            goal_threshold=0.5,
            goal_detection_threshold=4.0,
            **config_dict,
        )
        ant_program_env.reset()
        ant_program_env.force_execution = True

        # evaluation
        fail_num = 0
        total_reward = 0
        for exp_index in tqdm(range(1000)):
            ant_program_env.reset()
            ant_program.execute(ant_program_env)
            reward = ant_program_env.check_reward()
            if reward < 1:
                fail_num += 1
            total_reward += reward
            if fail_num > 50:
                break
        
        if fail_num > 50:
            log_and_print('evaluate on {} and fail'.format(task))
        else:
            log_and_print('evaluate on {} and get reward {}/1000'.format(task, total_reward))
        task_solve += total_reward >= 950
    
    return task_solve

def build_config(args):
    # general
    args.store_path = 'store/mcts_test/{}'.format(''.join(args.tasks.split(',')))

    # search
    args.search_seed_list = ','.join([str(1000 * exp_id) for exp_id in range(args.num_exps)])
    if args.num_exps > 5:
        args.support_seed_list = args.search_seed_list

def do_search(args):
    # initialize task
    task_list = args.tasks.split(',')
    assert len(task_list) > 0
    store_path = args.store_path
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    
    # other init
    eval_num = args.eval_num
    reward_list = []
    step_list = []
    random_seed_list = [int(seed) for seed in args.search_seed_list.split(',')]
    more_seeds = [int(seed) for seed in args.support_seed_list.split(',')]
    eval_seeds = [10000 + 1000 * i for i in range(eval_num)]

    # do search
    for seed_id, random_seed in enumerate(random_seed_list):
        print('currently random seed {}'.format(random_seed))
        reward_list.append([])
        step_list.append([])

        # set seed
        seed = random_seed
        cur_more_seeds = copy.deepcopy(more_seeds)
        if random_seed in cur_more_seeds:
            cur_more_seeds.pop(cur_more_seeds.index(random_seed))

        # reset random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        print('currently search seed {}'.format(seed))
        example_program = gt_structure()

        # search for each task
        start_time_step = 0
        for task in task_list:
            init_logging(store_path, 'log_{}_{}_{}.txt'.format(task, seed, random_seed))

            node = Node(sketch=example_program, task=task, 
                seed=seed, more_seeds=cur_more_seeds, eval_seeds=eval_seeds, 
                max_search_iter=500, max_structural_cost=20, shuffle_actions=True, found_one=True, prob_mode=False,
                sub_goals=[1.0])

            start_time = time.time()
            _, _, timesteps = node.search()
            total_time = time.time() - start_time
            log_and_print('time: {}'.format(total_time))

            # new program
            example_program = node.candidates['success'][0][1]
            example_program.reset()
            example_reward = eval_program(copy.deepcopy(example_program))
            reward_list[-1] += [example_reward for _ in timesteps]
            step_list[-1] += (np.array(timesteps) + start_time_step).tolist()
            start_time_step += timesteps[-1]

            # reset program after evaluation
            example_program.reset()
            example_program.reset_c_touch()

    # plot
    np.save('store/test_reward.npy', reward_list)
    np.save('store/test_step.npy', step_list)
    var_plot(reward_list, step_list, fig_path='{}/reward.pdf'.format(store_path))

if __name__ == "__main__":
    args = get_parse()
    build_config(args)

    do_search(args)