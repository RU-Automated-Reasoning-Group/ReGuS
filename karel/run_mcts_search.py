from mcts.MCTS_search_tree import SearchTree, SearchNode
from mcts.search_alg import SearchAlg
from dsl_karel import *
from utils.logging import init_logging, log_and_print
from parser_lib import get_karel_parser

import time
import random
import numpy as np

def get_param(args):
    # more select
    if args.more_seed is None:
        all_seeds = [78000, 85000, 71000, 0, 10000, 14000, 15000, 17000, 18000, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    else:
        all_seeds = [int(seed) for seed in args.more_seed.split(',')]
    
    # sub goals
    goals = [float(g) for g in args.sub_goals.split(',')]

    return all_seeds, goals

def do_search(args, search_steps, search_iter, task='topOff', \
              seed=123, more_seeds=[], score_C=1., cost_ratio=.5, goals=[1.]):
    # create root 
    root_sketch = Program()
    root = SearchNode(None, root_sketch, 0, False)
    root.visited = True

    # create alg
    alg = SearchAlg(task=task, seed=seed, \
                    more_seeds=more_seeds, \
                    max_search_iter=search_iter, max_structural_cost=20, \
                    structural_weight=args.stru_weight, shuffle_actions=True, \
                    goals=goals)
    log_and_print('#####################')
    log_and_print('for task {} with search seed {}'.format(task, seed))
    log_and_print('#####################')

    # create search tree
    mcts_tree = SearchTree(root, alg, score_C=score_C, cost_ratio=cost_ratio)
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

    return best_reward, total_time, cur_step


if __name__ == '__main__':
    # get parameters
    args = get_karel_parser()
    all_seeds, goals = get_param(args)

    # set random seed
    seed = args.search_seed
    random.seed(seed)
    np.random.seed(seed)

    # init
    task = args.task_name
    search_iter = args.search_iter

    init_logging('store/mcts_test', 'log_{}_{}_mcts.txt'.format(task, seed))
    log_and_print('currently search seed {}'.format(seed))

    best_reward, total_time, total_step = \
        do_search(1000, search_iter, task, seed, more_seeds=all_seeds, score_C=0.2, cost_ratio=.2)

    log_and_print('reward: {}\n time cost: {}\n all steps: {}'.format(best_reward, total_time, total_step))