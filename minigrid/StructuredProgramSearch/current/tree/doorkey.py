import copy
import pdb
import random

import dsl
import gym_minigrid.dsl
import numpy as np
from gym_minigrid.robot import MiniGridRobot

import search


def get_function(object):
    if object == 'key':
        obj_on_right = 'key_on_right'
        obj_present = 'key_present'
    elif object == 'door':
        obj_on_right = 'door_on_right'
        obj_present = 'door_present'

    fn = dsl.Program()
    
    first_while = dsl.WHILE()
    first_while.cond[0] = gym_minigrid.dsl.k_cond(
        False,
        gym_minigrid.dsl.k_cond_without_not('front_is_clear')
    )
    first_while.stmts[0] = dsl.ACTION(
        gym_minigrid.dsl.k_action('move')
    )

    fn.stmts[0] = first_while
    fn.stmts.insert(1, dsl.ACTION(
        gym_minigrid.dsl.k_action('turn_right')
    ))

    second_while = dsl.WHILE()
    second_while.cond[0] = gym_minigrid.dsl.k_cond(
        True,
        gym_minigrid.dsl.k_cond_without_not(obj_on_right)
    )
    second_while.stmts[0] = dsl.IF(
        cond=gym_minigrid.dsl.k_cond(
            False,
            gym_minigrid.dsl.k_cond_without_not('left_is_clear')
        )
    )
    second_while.stmts[0].stmts[0] = dsl.ACTION(
        gym_minigrid.dsl.k_action('turn_left')
    )

    second_while.stmts.append(
        dsl.IF(
            cond = gym_minigrid.dsl.k_cond(
                True,
                gym_minigrid.dsl.k_cond_without_not('front_is_clear')
            )
        )
    )
    second_while.stmts[-1].stmts[0] = dsl.ACTION(
        gym_minigrid.dsl.k_action('turn_right')
    )

    second_while.stmts.append(
        dsl.ACTION(
            gym_minigrid.dsl.k_action('move')
        )
    )

    fn.stmts.insert(2, second_while)

    third_while = dsl.WHILE()
    third_while.cond[0] = gym_minigrid.dsl.k_cond(
        True,
        gym_minigrid.dsl.k_cond_without_not(obj_present)
    )
    third_while.stmts[0] = dsl.ACTION(
        gym_minigrid.dsl.k_action('move')
    )

    fn.stmts.insert(3, dsl.ACTION(
        gym_minigrid.dsl.k_action('turn_right')
    ))
    fn.stmts.insert(3, third_while)
    return fn

if __name__ == "__main__":
    random.seed(61)
    np.random.seed(61)

    # NOTE: for simplicity, not a tree right now
    program_db = []
    # pdb.set_trace()

    p = dsl.Program()
    program_db.append(p)

    _p = copy.deepcopy(program_db[0])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[1])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[2])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[3])
    program_db += _p.expand()

    for p in program_db:
        print(p)    

    example_program = program_db[9].expand()[0].expand()[2].expand()[0].expand()[0].expand()[0]
    print(example_program)
    pdb.set_trace()

    seed = 12
    more_seeds = []
    eval_seeds = [seed]

    robot = MiniGridRobot('MiniGrid-DoorKey-8x8-v0', seed)
    from matplotlib.pyplot import imsave
    imsave(f"door_keyseeds/{seed}.png", robot.env.render(mode='rgb_array'))
    exit()
    node = search.Node(sketch=example_program, task='MiniGrid-DoorKey-8x8-v0', seed=seed, more_seeds=more_seeds, eval_seeds=eval_seeds, max_search_iter=10000, max_structural_cost=30, shuffle_actions=True, found_one=True)
    node.search()