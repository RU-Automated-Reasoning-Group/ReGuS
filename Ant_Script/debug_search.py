import time
import copy
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

from dsl import *
# from search_karel_new import Node
from search_karel_new_while import Node

from utils.logging import init_logging, log_and_print

import pdb

def get_structural_cost(program):
    cost = 0
    program_str = str(program)
    for s in ACTION_DICT:
        cost += program_str.count(s)

    return cost

def get_perc_action(action):
    # general post abstract state
    post_abs_state = ABS_STATE()
    for s in post_abs_state.state:
        post_abs_state.update(s, 'DNC')
    action.post_abs_state = post_abs_state
    # general pre abstract state
    pre_abs_state = ABS_STATE()
    for s in pre_abs_state.state:
        pre_abs_state.update(s, 'DNC')
    action.abs_state = pre_abs_state

    return action

def gt_structure():
    while_1 = WHILE()
    while_1.cond = [COND_DICT['not(present_goal)']]
    while_1.stmts = [C()]

    program = Program()
    program.stmts = [while_1, C(), program.stmts[-1]]

    return program


def direct_debug():
    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)

    task = 'AntU'
    # task = 'AntFb'
    # task = 'AntMaze'
    load_path = None
    # load_path = 'store_fea/AntU_3000_prog_new_r_count_if.pkl'
    # load_path = 'store_fea/AntU_0_prog_new_r_count_if.pkl'
    # load_path = 'store_fea/AntU_prog_new.pkl'

    seed = random_seed
    init_logging('store/', 'log_{}_{}_while_new_r_count_if_spec.txt'.format(task, seed))
    print('log_{}_{}.txt'.format(task, seed))

    if load_path is None:
        example_program = gt_structure()
    else:
        with open(load_path, 'rb') as f:
            example_program = pickle.load(f)
        example_program = example_program[0][1]

    print(example_program)
    pdb.set_trace()

    # harvester
    more_seeds = [0, 1000, 2000, 3000, 4000]
    # eval_seeds = [10000 + 1000 * i for i in range(10)]
    eval_seeds = []
    if random_seed in more_seeds:
        more_seeds.pop(more_seeds.index(random_seed))

    # np.random.shuffle(more_seeds)
    # seed = 0 # 1 can get reward=1 at once
    node = Node(sketch=example_program, task=task, 
                seed=seed, more_seeds=more_seeds, eval_seeds=eval_seeds, 
                max_search_iter=10000, max_structural_cost=20, shuffle_actions=True, found_one=True, prob_mode=False,
                sub_goals=[1.0])

    pdb.set_trace()
    start_time = time.time()
    all_rewards = node.search()
    total_time = time.time() - start_time
    log_and_print('time: {}'.format(total_time))
    # pdb.set_trace()
    # exit()

    with open('store_fea/{}_{}_prog_new_r_count_if_spec.pkl'.format(task, random_seed), 'wb') as f:
        pickle.dump(node.candidates['success'], f)

    # plot
    np.save('store/{}_{}_rewards.npy'.format(task, random_seed), all_rewards)
    plt.figure()
    plt.plot(np.arange(len(all_rewards)), all_rewards)
    plt.savefig('store/figs/{}_rewards.pdf'.format(task))


def multi_debug():

    # all_seeds = [78000, 85000, 71000, 0, 10000, 14000, 15000, 17000, 18000, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    # all_seeds = [4000, 8000, 0, 71000, 85000, 9000, 5000, 2000, 78000, 6000, 3000]
    all_seeds = [85000, 0, 71000, 4000, 8000, 9000, 5000, 2000, 78000, 6000, 3000]
    # all_seeds = [85000, 0, 71000, 5000]
    # all_seeds = [85000, 0, 71000, 10000, 14000, 15000]
    result = {}
    # test_seeds = [85000, 18000, 1000]
    # test_seeds = all_seeds[1:4]
    test_seeds = [85000, 71000, 0]
    # test_seeds = all_seeds
    # all_seeds = all_seeds[1:4]
    # all_seeds = all_seeds[1:4]
    all_seeds = all_seeds
    all_rewards = []

    for seed in test_seeds:
        random_seed = seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # task = 'harvester'
        # task = 'seeder'
        task = 'doorkey'
        # seed = 3
        seed = random_seed
        init_logging('store/karel_log', 'log_{}_{}_subgoals_cost.2.txt'.format(task, seed))
        print('log_{}_{}.txt'.format(task, seed))

        # example_program = gt_structure()
        example_program = gt_structure_2()
        # example_program = gt_structure_3()

        print(example_program)
        pdb.set_trace()

        eval_seeds = [10000 + 1000 * i for i in range(10)]

        # np.random.shuffle(more_seeds)
        # seed = 0 # 1 can get reward=1 at once
        more_seeds = copy.deepcopy(all_seeds)
        more_seeds.pop(more_seeds.index(seed))
        node = Node(sketch=example_program, task=task, 
                    seed=seed, more_seeds=more_seeds, eval_seeds=eval_seeds, 
                    max_search_iter=8000, max_structural_cost=20, shuffle_actions=True, found_one=True, prob_mode=False, 
                    sub_goals=[0.5, 1.0])

        node.robot_store[seed].draw()

        start_time = time.time()
        cur_rewards, cur_eval_rewards, cur_timesteps = node.search()
        total_time = time.time() - start_time
        log_and_print('time: {}'.format(total_time))
        pdb.set_trace()
        
        if len(node.candidates['success']) > 0:
            result[seed] = [1, cur_eval_rewards[-1], cur_timesteps[-1]]
        else:
            result[seed] = [0, cur_eval_rewards[-1], cur_timesteps[-1]]
    
        # plot
        all_rewards.append([cur_rewards, cur_eval_rewards, cur_timesteps])
        np.save('store/karel_log/{}_{}seeds_rewards_steps_less.npy'.format(task, len(all_seeds)), all_rewards)
    # plt.figure()
    # plt.plot(np.arange(len(all_rewards)), all_rewards)
    # plt.savefig('store/karel_log/figs/{}_rewards.pdf'.format(task))

    print(result)
    pdb.set_trace()
    print('o?')


if __name__ == "__main__":
    direct_debug()
    # multi_debug()