import random
import time

import numpy as np
from dsl import *

from mcts.A_star_search_tree import SearchNode, SearchTree
from mcts.search_alg import SearchAlg
from utils.logging import init_logging, log_and_print


def test(search_steps, search_iter, task='topOff', seed=123, more_seeds=[], eval_seeds=[]):
    # create root 
    root_sketch = Program()
    root = SearchNode(None, root_sketch, 0, False)
    root.f_cost = 2

    # create alg
    alg = SearchAlg(task=task, seed=seed, \
                    more_seeds=more_seeds, \
                    eval_seeds=eval_seeds, \
                    max_search_iter=search_iter, max_structural_cost=20, shuffle_actions=True)
    log_and_print('#####################')
    log_and_print('for task {} with search seed {}'.format(task, seed))
    log_and_print('#####################')

    # create search tree
    astar_tree = SearchTree(root, alg)
    astar_tree.display_all()
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
        # expand children for one step
        leaf_node, reward, result_sketches, pass_steps = astar_tree.expand()
        if reward > best_reward:
            best_reward = reward
            best_sketch = result_sketches

        log_and_print('Total Time Used: {}'.format(time.time() - start_time))
        log_and_print('Current Reward: {}'.format(reward))
        astar_tree.display_all()
        store_results[cur_step] = reward
        cur_step += pass_steps
        log_and_print('')
        # pdb.set_trace()

        if reward == 1:
            break

    # print result
    total_time = time.time() - start_time
    log_and_print('Total Step Used: {}'.format(cur_step))
    log_and_print('Total Time Used: {}'.format(total_time))
    log_and_print('best reward: {}'.format(best_reward))
    if best_reward == 1:
        for s_prog in best_sketch['success']:
            log_and_print(str(s_prog[1]))
    # pdb.set_trace()
    print('haha')

    return best_reward, total_time, cur_step

if __name__ == '__main__':
    np.random.seed(4)
    random.seed(4)
    task = "MiniGrid-MultiRoom-N6-v0" 
    search_iter = 5000
    seed = 58
    more_seeds = [4000, 2, 12]
    eval_seeds = [4000, 2, 12, 13, 16]
    from gym_minigrid.robot import MiniGridRobot
    a = MiniGridRobot(task, seed)
    a.env.render()
    init_logging('store/astar_test', 'log_{}_{}_{}.txt'.format(task, seed, ""))
    best_reward, total_time, total_step = test(1000, search_iter, task, seed, more_seeds=more_seeds, eval_seeds=eval_seeds)
    # eval_num = 10
    # reward_list = []
    # time_list = []
    # step_list = []
    # random_seed_list = [0, 2000, 3000, 4000, 6000]
    # eval_seeds = [10000, 11000, 14000, 15000, 17000]
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
    #     # task = 'harvester'
    #     # task = 'fourCorners'
    #     # task = 'topOffPick'
    #     # search_iter = 3000
    #     task = "MiniGrid-MultiRoom-N6-v0"
    #     search_iter = 3000
    #     # search_iter = 1000
    #     # search_iter = 1
    #     seed = 6000
    #     print('currently search seed {}'.format(seed))
    #     if seed != 6000:
    #         continue
    #     from gym_minigrid.robot import MiniGridRobot
    #     a = MiniGridRobot(task, 6000)
    #     a.env.render()
    #     init_logging('store/astar_test', 'log_{}_{}_{}.txt'.format(task, seed, random_seed))
    #     best_reward, total_time, total_step = test(1000, search_iter, task, seed, more_seeds=more_seeds, eval_seeds=eval_seeds)

    #     reward_list.append(best_reward)
    #     time_list.append(total_time)
    #     step_list.append(total_step)

    # print('avg reward {} / std reward {}'.format(np.mean(reward_list), np.std(reward_list)))
    # print('avg time {} / std time {}'.format(np.mean(time_list), np.std(time_list)))
    # print('avg step {} / std step {}'.format(np.mean(step_list), np.std(step_list)))