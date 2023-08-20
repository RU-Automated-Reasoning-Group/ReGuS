# NOTE: node in the tree, contains a sketch and a queue for searching
import copy
import pdb
import random

import numpy as np
from dsl import *

from search import Node
from utils.logging import init_logging


def get_structural_cost(program):
    cost = 0
    program_str = str(program)
    for s in ACTION_DICT:
        cost += program_str.count(s)

    return cost


if __name__ == "__main__":

    random.seed(123)
    np.random.seed(123)

    task = 'topOffPick'
    # task = 'stairClimber'
    # task='cleanHouse'
    # task = 'topOff'
    # task = 'randomMaze'
    # task = 'fourCorners'
    # task = 'harvester'
    seed = 4
    # seed = 888
    init_logging('store/new_search_log', 'log_{}_{}.txt'.format(task, seed))

    # NOTE: for simplicity, not a tree right now
    program_db = []

    p = Program()
    program_db.append(p)

    _p = copy.deepcopy(program_db[0])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[1])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[2])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[3])
    program_db += _p.expand()

    # for general case
    # TODO: try to find simpler program for seed = 0
    # program : WHILE(not (markers_present)) { C } ; S ; END
    # example_program = copy.deepcopy(program_db[9]).expand()[0]
    # example_program = example_program.expand()[1]
    _p = copy.deepcopy(program_db[8])
    new_lib = _p.expand()
    # _p = copy.deepcopy(new_lib[7])
    # new_lib = _p.expand()
    # _p = copy.deepcopy(new_lib[0])
    # new_lib = _p.expand()
    # _p = copy.deepcopy(new_lib[1])
    # new_lib = _p.expand()
    # _p = copy.deepcopy(new_lib[3])
    # new_lib = _p.expand()
    # _p = copy.deepcopy(new_lib[0])
    # new_lib = _p.expand()
    _p = copy.deepcopy(new_lib[0])
    new_lib = _p.expand()
    _p = copy.deepcopy(new_lib[0])
    new_lib = _p.expand()

    # pdb.set_trace()
    example_program = _p
    # example_program = copy.deepcopy(new_lib[0]).expand()[0]
    # example_program = copy.deepcopy(program_db[6]).expand()[0]
    # example_program  = copy.deepcopy(new_lib[0]).expand()[0]
    # example_program = new_lib[1]
    print(example_program)
    more_seeds = [999, 123, 666, 546, 11, 4372185, 6431, 888, 1, 2, 3, 4, 5, 0]
    # seed = 0 # 1 can get reward=1 at once
    node = Node(sketch=example_program, task=task, seed=seed, more_seeds=more_seeds, max_search_iter=4000, max_structural_cost=20, shuffle_actions=False)
    node.robot_store[seed].draw()
    # print("")
    # for e in more_seeds:
    #     print("seed:{}".format(e))
    #     node = Node(sketch=example_program, task=task, seed=e, more_seeds=[], max_search_iter=4000, max_structural_cost=20, shuffle_actions=False)
    #     node.robot_store[e].draw()
    #     print("")
    pdb.set_trace()
    node.search()
    exit()

    # TODO: cannot solve consecutive cases, e.g., 12
    # task    : topOff
    # program : WHILE(front_is_clear) { C } ; S ; END
    # example_program = program_db[-2]
    # more_seeds = [999, 123, 666, 11, 4372185, 6431, 888, 1, 2, 3, 4, 5, 546, 12]
    # # more_seeds = []
    # # seed = 37 # 0/123/12/546/37, etc. 
    # node = Node(sketch=example_program, task='topOff', seed=seed, more_seeds=more_seeds, max_search_iter=1000, max_structural_cost=20, shuffle_actions=True)
    # node.robot.draw()
    # pdb.set_trace()
    # node.search()
    # exit()

    # TODO: try to merge consecutive actions
    # task    : stairClimber
    # program : WHILE(not (front_is_clear)) { C } ; S ; END
    # example_program = program_db[7].expand()[0]
    # more_seeds = [1, 123, 432, 84314, 73]
    # seed = 321 # 1/321/123, etc.
    # node = Node(sketch=example_program, task='stairClimber', seed=seed, more_seeds=more_seeds, max_search_iter=100, max_structural_cost=20, shuffle_actions=True)
    # node.search()
    # exit()
    
    # NOTE: okay for now
    # task    : fourCorner
    # program : WHILE(not (markers_present)) { WHILE(front_is_clear) { C } ; C } ; S ; END
    # example_program = copy.deepcopy(program_db[-3]).expand()[-1].expand()[0].expand()[0].expand()[0]
    # node = Node(sketch=example_program, task='fourCorners', seed=123, more_seeds=[], max_search_iter=100, max_structural_cost=20, shuffle_actions=True)
    # node.search()
    # exit()

    # TODO: try to find simpler program for seed = 0
    # task    : randomMaze
    # program : WHILE(not (markers_present)) { C } ; S ; END
    # example_program = copy.deepcopy(program_db[9]).expand()[0]
    # # example_program = copy.deepcopy(program_db[6]).expand()[0]
    # more_seeds = [999, 123, 666, 546, 11, 4372185, 6431, 888, 1, 2, 3, 4, 5, 0]
    # # seed = 0 # 1 can get reward=1 at once
    # node = Node(sketch=example_program, task=task, seed=seed, more_seeds=more_seeds, max_search_iter=1000, max_structural_cost=20, shuffle_actions=False)
    # node.robot.draw()
    # # for e in more_seeds:
    # #     node = Node(sketch=example_program, task=task, seed=e, more_seeds=more_seeds, max_search_iter=1000, max_structural_cost=20, shuffle_actions=False)
    # #     node.robot.draw()
    # #     print()
    # pdb.set_trace()
    # node.search()
    # exit()

