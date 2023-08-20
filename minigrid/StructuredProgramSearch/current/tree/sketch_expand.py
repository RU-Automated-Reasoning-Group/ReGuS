import copy
import queue
import random
import time
from queue import PriorityQueue

import numpy as np
from dsl import *

from karel.robot import KarelRobot
from search import Node

if __name__ == "__main__":

    random.seed(123)
    np.random.seed(123)

    # NOTE: sketch expansion
    program_db = []

    p = Program()  # start with [ S ; END ], the simplest program
    program_db.append(p)

    # NOTE: important, before expansion, make sure to deepcopy to backup,
    #       otherwise the original program would be affected
    _p = copy.deepcopy(program_db[0])
    
    # NOTE: Program.expand() would apply production rules to expand the sketch
    #       and reture a list which contains two new sketches
    # NOTE: Program.expand() will only expand the leftmost non-terminals, like B or S,
    #       if more than one B or S exist, it still can only apply a production rule once
    p_list = _p.expand()
    for sketch in p_list:
        print(sketch)
    program_db += p_list

    _p = copy.deepcopy(program_db[1])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[2])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[3])
    program_db += _p.expand()


    for sketch in program_db:
        print(sketch)

    # NOTE: note that not every sketch can be used to search,
    #       we must consolidate the B to a specific condition,
    #       and convert S in any loop to an C, so that we can do the search

    # example 1：WHILE(front_is_clear) { S } ; S ; END ----> This is NOT complete (or you can say not executable)
    # example 2: WHILE(not (markers_present)) { C } ; S ; END ----> This is complete, even if there is an S after the WHILE
    #            I will write an function to determine which one is complete (executable)

    # NOTE: node() contains a sketch
    #       you can let more_seeds to be empty, more_seeds=[]
    #       shuffle_actions means when expand the actions, the order of expansion is randomized
    #       *I will support random order expansion of sketch later
    example_program = program_db[-2]
    node = Node(sketch=example_program, task='topOff', seed=123, more_seeds=[999, 321], max_search_iter=100, max_structural_cost=20, shuffle_actions=True)
    
    # NOTE: Node.search() is used to search programs which can solve the input task,
    #       after searching complete, programs would be stored in Node.candidates (type: dict)
    # node.search()
    # print('[debug]', node.candidates['success'])
    # print('[debug]', node.candidates['failed'])
    # print('[debug]', node.candidates['no_fuel'])
    # print('[debug]', node.candidates['complete'])

    # NOTE: currently, 'topOff', 'stairClimber' and 'fourCorner' can be solved well

    # TODO: sequential C situations
    print('[try to handle sequential C]')
    #print(program_db[-2]) # WHILE(front_is_clear) { C } ; S ; END
    #print(program_db[-2].expand()[0])  # WHILE(front_is_clear) { C } ; C ; END
    #print(program_db[-2].expand()[1])  # WHILE(front_is_clear) { C } ; WHILE(left_is_clear) { S } ; S ; END

    # WHILE(front_is_clear) { C } ; C ; END
    _p = copy.deepcopy(program_db[-2])
    example_program = _p.expand()[0]
    c_stmts, c_index = example_program.find_c_cond()
    print('[1]', c_stmts, c_index)

    # WHILE(front_is_clear) { move } ; C ; END
    _p = copy.deepcopy(program_db[-2])
    example_program = _p.expand()[0]
    example_program.stmts[0].stmts[0] = ACTION(k_action('move'))
    c_stmts, c_index = example_program.find_c_cond()
    if c_stmts is None:
        c_stmts, c_index = example_program.find_seq_c()
    print('[2]', c_stmts, c_index)
    print(example_program)
    for _p in example_program.expand_actions():
        print(_p)
