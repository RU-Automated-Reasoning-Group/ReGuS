import pdb
import random
import time

import numpy as np
import torch
import minigrid_base_dsl

from mcts.MCTS_search_tree import SearchNode, SearchTree
from mcts.search_alg import SearchAlg
from utils.logging import init_logging, log_and_print


def mcts_search(search_steps, search_iter, task='topOff', seed=123, more_seeds=[], eval_seeds=[], lib_actions=None):
    # create root 
    root_sketch = minigrid_base_dsl.Program()
    root = SearchNode(None, root_sketch, 0, False, lib_actions=lib_actions)
    root.visited = True

    # create alg
    alg = SearchAlg(task=task, seed=seed, \
                    more_seeds=more_seeds, \
                    eval_seeds = eval_seeds, \
                    max_search_iter=search_iter, max_structural_cost=20, shuffle_actions=True)
    log_and_print('#####################')
    log_and_print('for task {} with search seed {}'.format(task, seed))
    log_and_print('#####################')

    # create search tree
    mcts_tree = SearchTree(root, alg)
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
            log_and_print(s_prog[1].to_string_verbose())
    # pdb.set_trace()
    print('haha')

    return best_reward, total_time, cur_step


if __name__ == '__main__':
    random_seed = 13
    random.seed(random_seed)
    np.random.seed(random_seed)
    seed = 5
    more_seeds = [i for i in range(0, 500)]
    eval_seeds = [i for i in range(501, 1000)]
    # eval_seeds = [2, 14, 18, 24, 29, 33, 44, 47, 67, 70, 76, 96, 105, 112, 113, 136, 150, 160, 170, 183, 199, 205, 218, 227, 229, 230, 235, 262, 280, 287, 297, 300, 306, 312, 314, 316, 318, 325, 333, 335, 337, 351, 352, 364, 366, 367, 370, 373, 379, 385, 388, 402, 408, 416, 431, 432, 433, 434, 442, 457, 460, 461, 476, 481, 483, 485, 498, 501, 504, 507, 515, 517, 527, 534, 542, 544, 574, 579, 586, 591, 595, 600, 605, 608, 615, 617, 619, 621, 622, 623, 624, 626, 627, 629, 632, 644, 645, 667, 668, 680, 683, 684, 691, 702, 703, 706, 725, 727, 729, 765, 770, 788, 797, 809, 814, 821, 830, 836, 838, 845, 846, 847, 873, 877, 886, 897, 906, 913, 930, 932, 933, 936, 940, 977, 982, 992, 1010, 1014, 1016, 1018, 1029, 1033, 1039, 1043, 1045, 1052, 1060, 1073, 1075, 1078, 1082, 1084, 1088, 1093, 1098]
    # more_seeds = eval_seeds[0:50]
    # eval_seeds = [0]
    # for i in range(0, 10000):
    #     from gym_minigrid.robot import MiniGridRobot
    #     robot = MiniGridRobot('MiniGrid-UnLockedRoom-v0', i)
    #     env = robot.env
    #     left_pos = env.agent_pos + (-1 * env.right_vec)
    #     left_cell = env.grid.get(*left_pos)
    #     if left_cell is not None and left_cell.type == 'wall':
    #         eval_seeds.append(i)
    #         if len(eval_seeds) >= 1000:
    #             break
        # pdb.set_trace()
    # print(eval_seeds)
    # more_seeds = eval_seeds[100: 130]
    task = 'MiniGrid-Empty-Random-8x8-v0'
    init_logging('store/mcts_test', 'log_{}_{}_{}.txt'.format(task, seed, random_seed))
    log_and_print(eval_seeds)
    test(10000, 200, task, seed, more_seeds, eval_seeds)
    # eval_num = 10
    # reward_list = []
    # time_list = []
    # step_list = []
    # random_seed_list = [0, 1000, 2000, 3000, 4000]
    # eval_seeds = [10000 + 1000 * i for i in range(eval_num)]
    # for seed_id, random_seed in enumerate(random_seed_list):
    #     print('currently random seed {}'.format(random_seed))

    #     # more_seeds = [999, 123, 666, 546, 11, 4372185, 6431, 888, 0, 1, 2, 3, 4, 5]
    #     more_seeds = [0, 1000, 2000, 3000, 4000]
    #     seed = more_seeds[seed_id % len(more_seeds)]

    #     # reset random seed
    #     random.seed(random_seed)
    #     np.random.seed(random_seed)
    #     np.random.shuffle(more_seeds)

    #     # task = 'cleanHouse'
    #     # task = 'stairClimber'
    #     # task = 'topOff'
    #     # task = 'randomMaze'
    #     task = 'harvester'
    #     # task = 'fourCorners'
    #     # task = 'topOffPick'
    #     # search_iter = 3000
    #     # search_iter = 2000
    #     search_iter = 1000
    #     # search_iter = 1

    #     print('currently search seed {}'.format(seed))

    #     init_logging('store/mcts_test', 'log_{}_{}_{}.txt'.format(task, seed, random_seed))
    #     best_reward, total_time, total_step = test(1000, search_iter, task, seed, more_seeds=more_seeds, eval_seeds=eval_seeds)

    #     reward_list.append(best_reward)
    #     time_list.append(total_time)
    #     step_list.append(total_step)

    #     break

    # print('avg reward {} / std reward {}'.format(np.mean(reward_list), np.std(reward_list)))
    # print('avg time {} / std time {}'.format(np.mean(time_list), np.std(time_list)))
    # print('avg step {} / std step {}'.format(np.mean(step_list), np.std(step_list)))