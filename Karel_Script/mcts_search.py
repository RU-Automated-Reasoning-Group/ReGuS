from mcts.MCTS_search_tree import SearchTree, SearchNode
from mcts.search_alg import SearchAlg
# from dsl import *
from dsl_karel_new import *
from utils.logging import init_logging, log_and_print
from utils.parse_lib import get_parse

import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

import pdb

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
        for step_id, need_step in enumerate(entire_time_steps):
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


def test(search_steps, search_iter, task='topOff', seed=123, more_seeds=[], eval_seeds=[], score_C=1., cost_ratio=.5, sub_goals=[1.0], cost_w=0.04):
    # create root 
    if task == 'doorkey':
        root_sketch = Program(set_unit=True)
        sub_goals = [0.5, 1.0]
        cond_cost_use = True
    if task == 'fourCornersRandom':
        root_sketch = Program(set_unit=True)
        cond_cost_use = False
    if task == 'cleanHouse':
        root_sketch = Program()
        cond_cost_use = True
    else:
        root_sketch = Program()
        cond_cost_use = False
    root = SearchNode(None, root_sketch, 0, False)
    root.visited = True

    # create alg
    alg = SearchAlg(task=task, seed=seed, \
                    more_seeds=more_seeds, \
                    eval_seeds = eval_seeds, \
                    max_search_iter=search_iter, max_structural_cost=20, shuffle_actions=True, sub_goals=sub_goals, cost_w=cost_w)
    log_and_print('#####################')
    log_and_print('for task {} with search seed {}'.format(task, seed))
    log_and_print('#####################')

    # create search tree
    mcts_tree = SearchTree(root, alg, score_C=score_C, cost_ratio=cost_ratio, search_iter_limits=search_iter, cond_cost_use=cond_cost_use)
    mcts_tree.display_all()
    print()

    # time
    start_time = time.time()

    # do mcts search
    cur_step = 0
    best_reward = 0
    best_sketch = None
    store_results = {}
    while cur_step < search_steps:
        log_and_print('Current Step: {}'.format(cur_step))
        # traverse
        result_node = mcts_tree.traverse()
        result_node.visited = True
        # rollout
        leaf_node, reward, result_sketches, pass_step = mcts_tree.rollout(result_node)
        if reward > best_reward:
            best_reward = reward
            best_sketch = result_sketches
        # backprop
        mcts_tree.backprop(result_node, reward)

        log_and_print('Total Time Used: {}'.format(time.time() - start_time))
        log_and_print('Current Reward: {}'.format(reward))
        mcts_tree.display_all()
        store_results[cur_step] = reward
        cur_step += pass_step

        if reward == 1:
            break

        if time.time() - start_time > 7200:
            break

    # decode rewards
    if best_reward != 1:
        if best_reward == 0.8:
            best_reward = 0
        elif best_reward >= 0.5:
            best_reward = (best_reward-0.5) / 0.3
        else:
            best_reward = best_reward / 0.3

    # print result
    total_time = time.time() - start_time
    log_and_print('Total Step Used: {}'.format(cur_step))
    log_and_print('Total Time Used: {}'.format(total_time))
    log_and_print('best reward: {}'.format(best_reward))
    if best_reward == 1:
        for s_prog in best_sketch['success']:
            log_and_print(str(s_prog[1]))

    return best_reward, total_time, cur_step, [alg.total_iter, alg.total_reward, alg.total_eval_reward]

def build_config(args):
    # general
    args.store_path = 'store/mcts_test/karel_log/{}'.format(args.task)
    if args.task in ['seeder', 'doorkey']:
        args.search_iter = '600,3500'
    else:
        args.search_iter = '500'

    # search
    args.search_seed_list = ','.join([str(1000 * exp_id) for exp_id in range(args.num_exps)])
    if args.num_exps > 5:
        args.support_seed_list = args.search_seed_list
    if args.task == 'harvester':
        args.support_seed_list = ''
    if args.task in ['seeder', 'doorkey']:
        args.cost_w = 0.2
    else:
        args.cost_w = 0.04

if __name__ == '__main__':
    args = get_parse()
    build_config(args)

    # initialize task from argument
    task = args.task
    search_iter = [int(iter_num) for iter_num in args.search_iter.split(',')]
    store_path = args.store_path
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # other init
    eval_num = args.eval_num
    reward_list = []
    time_list = []
    step_list = []
    time_step_list = []
    random_seed_list = [int(seed) for seed in args.search_seed_list.split(',')]
    if len(args.support_seed_list) > 0:
        more_seeds = [int(seed) for seed in args.support_seed_list.split(',')]
    else:
        more_seeds = []
    eval_seeds = [10000 + 1000 * i for i in range(eval_num)]

    # do search
    for seed_id, random_seed in enumerate(random_seed_list):
        print('currently random seed {}'.format(random_seed))

        seed = random_seed
        cur_more_seeds = copy.deepcopy(more_seeds)
        if random_seed in cur_more_seeds:
            cur_more_seeds.pop(cur_more_seeds.index(random_seed))

        # reset random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        np.random.shuffle(cur_more_seeds)

        print('currently search seed {}'.format(seed))

        init_logging(store_path, 'log_{}_{}_{}.txt'.format(task, seed, random_seed))
        best_reward, total_time, total_step, total_time_step_results = \
            test(1000, search_iter, task, seed, more_seeds=cur_more_seeds, eval_seeds=eval_seeds, score_C=0.2, cost_ratio=.2, cost_w=args.cost_w)

        reward_list.append(best_reward)
        time_list.append(total_time)
        step_list.append(total_step)
        time_step_list.append(total_time_step_results)
        np.save('{}/{}_rewards_steps.npy'.format(store_path, task), time_step_list)

    print('avg reward {} / std reward {}'.format(np.mean(reward_list), np.std(reward_list)))
    print('avg time {} / std time {}'.format(np.mean(time_list), np.std(time_list)))
    print('avg step {} / std step {}'.format(np.mean(step_list), np.std(step_list)))

    var_plot(all_time_steps=[each_list[0] for each_list in time_step_list], 
             all_rewards=[each_list[1] for each_list in time_step_list], 
             fig_path='{}/{}_reward.pdf'.format(store_path, task))