import time
import copy
import random
import numpy as np

from dsl_karel import *
from search_karel import Node
from utils.logging import init_logging, log_and_print
from parser_lib import get_karel_parser

import pdb

def get_structure(args):
    program = Program()
    if args.task_name == 'randomMaze' or args.task_name == 'fourCorners':
        while_1 = WHILE()
        while_1.cond = [COND_DICT['not(markers_present)']]
        while_1.stmts = [C()]

        program.stmts = [while_1, C(), program.stmts[-1]]

    elif args.task_name == 'topOff':
        while_1 = WHILE()
        while_1.cond = [COND_DICT['front_is_clear']]
        while_1.stmts = [C()]

        program.stmts = [while_1, C(), program.stmts[-1]]

    elif args.task_name == 'stairClimber':
        while_1 = WHILE()
        while_1.cond = [COND_DICT['not(front_is_clear)']]
        while_1.stmts = [C()]

        program.stmts = [while_1, C(), program.stmts[-1]]
    
    elif args.task_name == 'cleanHouse':
        while_1 = WHILE()
        while_1.cond = [COND_DICT['not(markers_present)']]
        while_1.stmts = [C()]

        while_2 = WHILE()
        while_2.cond = [COND_DICT['not(markers_present)']]
        while_2.stmts = [while_1, C()]

        program.stmts = [while_2, C(), program.stmts[-1]]

    elif args.task_name == 'harvester':
        while_1 = WHILE()
        while_1.cond = [COND_DICT['left_is_clear']]
        while_1.stmts = [C()]

        program.stmts = [while_1, C(), program.stmts[-1]]

    elif args.task_name == 'seeder':
        while_1 = WHILE()
        while_1.cond = [COND_DICT['not(markers_present)']]
        while_1.stmts = [C()]

        while_2 = WHILE()
        while_2.cond = [COND_DICT['not(markers_present)']]
        while_2.stmts = [while_1, C()]

        program.stmts = [while_2, C(), program.stmts[-1]]

    elif args.task_name == 'doorkey':
        while_1 = WHILE()
        while_1.cond = [COND_DICT['not(markers_present)']]
        while_1.stmts = [C()]

        while_2 = WHILE()
        while_2.cond = [COND_DICT['not(markers_present)']]
        while_2.stmts = [C()]

        program = Program()
        program.stmts = [while_1, C(act_num=1), while_2, C(), program.stmts[-1]]

    return program

def get_param(args):
    # more select
    if args.more_seed is None:
        all_seeds = [78000, 85000, 71000, 0, 10000, 14000, 15000, 17000, 18000, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    else:
        all_seeds = [int(seed) for seed in args.more_seed.split(',')]
    
    # sub goals
    goals = [float(g) for g in args.sub_goals.split(',')]

    return all_seeds, goals

def do_synthesize(args):
    # get parameters
    all_seeds, goals = get_param(args)
    seed = args.search_seed

    # set seed
    random.seed(seed)
    np.random.seed(seed)

    # init
    task = args.task_name
    init_logging('store/karel_log', 'log_{}_{}.txt'.format(task, seed))

    # get sketch
    sketch = get_structure(args)
    log_and_print('search based on sketch:\n {}'.format(sketch))

    # seed = 0 # 1 can get reward=1 at once
    more_seeds = copy.deepcopy(all_seeds)
    if seed in more_seeds:
        more_seeds.pop(more_seeds.index(seed))

    node = Node(sketch=sketch, task=task, 
                seed=seed, more_seeds=more_seeds, 
                max_search_iter=args.search_iter, max_structural_cost=args.max_stru_cost, 
                structural_weight=args.stru_weight,
                shuffle_actions=True, found_one=True, 
                sub_goals=goals)

    node.robot_store[seed].draw()

    start_time = time.time()
    cur_rewards, cur_timesteps = node.search()
    total_time = time.time() - start_time
    log_and_print('time: {}'.format(total_time))
    
    if len(node.candidates['success']) > 0:
        result = 1
    else:
        result = 0

    log_and_print('search success: {}'.format(result==1))


if __name__ == "__main__":
    args = get_karel_parser()
    do_synthesize(args)