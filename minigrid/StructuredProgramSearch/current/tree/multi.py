import random
import time
from copy import deepcopy

import multiprocess as mp
import numpy as np
from dsl import *
from gym_minigrid.robot import MiniGridRobot

from mcts.A_star_search_tree import SearchNode, SearchTree
from mcts.search_alg import SearchAlg
from utils.logging import init_logging, log_and_print


def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):

    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
    parallel_runs = [pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts-1)
    pool.close()
    pool.terminate()
    pool.join()
    return results

def astar_search(search_steps, search_iter, task='topOff', seed=123, more_seeds=[], eval_seeds=[], logic_expr=None):
    # create root 
    root_sketch = Program()
    root = SearchNode(None, root_sketch, 0, False)
    root.f_cost = 2

    # create alg
    alg = SearchAlg(task=task, seed=seed, \
                    more_seeds=more_seeds, \
                    eval_seeds=eval_seeds, \
                    max_search_iter=search_iter, max_structural_cost=20, shuffle_actions=True, logic_expr=logic_expr)
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

        end_time = time.time()
        if end_time - start_time > 60:
            # last for 2 mins
            log_and_print('time out for seed: {}'.format(seed))
            return None

    # print result
    total_time = time.time() - start_time
    log_and_print('Total Step Used: {}'.format(cur_step))
    log_and_print('Total Time Used: {}'.format(total_time))
    log_and_print('best reward: {}'.format(best_reward))
    if best_reward == 1:
        return best_sketch['success'][0][1]
        for s_prog in best_sketch['success']:
            log_and_print(str(s_prog[1]))
    # pdb.set_trace()
    print('haha')

def eval_program(prog, task, all_seeds):
    num = 0
    abs_states_list = []
    for s in all_seeds:
        force_eval_robot = MiniGridRobot(task, seed=s)
        initial_abs_state = get_abs_state(force_eval_robot)
        force_eval_robot.force_execution = True
        prog.execute(force_eval_robot)
        prog.reset()
        force_eval_robot.force_execution = False
        if force_eval_robot.reward == 1:
            num += 1
            abs_states_list.append(initial_abs_state)
    return num, abs_states_list

class logic_expr():
    def __init__(self, abs_state_list):
        self.abs_state_list = copy.deepcopy(abs_state_list)
    
    def __call__(self, input_robot):
        input_abs_state = get_abs_state(input_robot)
        for state in self.abs_state_list:
            rst = True
            for k in state.state:
                if state.state[k] == 'F':
                    if state.state[k] != input_abs_state.state[k]:
                        rst = False

            if rst:
                # this abs_state matches the input abs_state
                return True
        # none of the state matches the input abs_state
        return False

def multi(task, all_seeds):
    # parameters for A* search
    search_steps = 1000
    search_iter = 3000

    # prepare inputs to each thread
    # input_dict_list = []
    # for s in all_seeds:
    #     input_dict = dict(search_steps=search_steps, search_iter=search_iter, task=task, seed=s, more_seeds=[], logic_expr=None)
    #     input_dict_list.append(input_dict)
    # results = _try_multiprocess(astar_search, input_dict_list, 4, 6000, 6000)
    p = astar_search(search_steps, search_iter, task, 4000, more_seeds=[], eval_seeds=[4000], logic_expr=None )
    print("p", p)
    results = [p]

    # evalute each found program
    print("+++++++++++++++++++++++++++++++++++++")
    print("results", results)
    lst = []
    for prog in results:
        if prog is not None:
            num_solved, abs_state_list = eval_program(prog, task, all_seeds)
            lst.append([prog, num_solved, abs_state_list])
    lst.sort(key=lambda x: x[1], reverse=True)
    print(lst)
    # this can also be made parallel
    for prog, n, abs_state_list in lst:
        # compile the logic expression
        print(n)
        print(prog)
        print(abs_state_list)

        abs_state_list = [get_abs_state(MiniGridRobot(task, 4000))]
        print(abs_state_list[0])
        expression = logic_expr(abs_state_list)

        # find corresponding p2
        p2 = astar_search(search_steps, search_iter, task, all_seeds[1], all_seeds, eval_seeds=all_seeds, logic_expr=expression)

        if p2 is not None:
            # found p2
            # print the combination of 2 program
            # pdb.set_trace()
            # robot = MiniGridRobot(task, 0)
            # robot.force_execution = True
            # p2.reset()
            # p2.execute_with_plot(robot)

            print(p2)
            print(prog)
            exit()

if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    task = 'MiniGrid-MultiRoom-N6-v0'
    robot = MiniGridRobot('MiniGrid-MultiRoom-N6-v0', 0)
    robot.env.render()
    # all_seeds = [0, 1, 2, 3, 4, 5]
    all_seeds = [4000, 12, 13, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    init_logging('store/astar_test', 'log_{}_{}.txt'.format(task, "all"))
    multi(task, all_seeds)