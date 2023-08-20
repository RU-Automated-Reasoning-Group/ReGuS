from sys import argv

import gym
import torch
import torch.nn as nn
import numpy as np

import gym_karel
from karel.karel import Karel
from karel.dsl import *
from karel.env import KarelRobotCleanHouse, KarelRobotHarvester, KarelRobotStairClimer, KarelRobotTopOff, KarelStatesLogger, KarelRobotEnv, KarelRobotRandomMaze, KarelRobotFourCorners


##############
# Simple Test
##############
def simple_test():

    K = KarelRobotEnv(env='randomMaze', seed=999)

    p = k_prog(
            k_stmt(
                k_stmt_stmt(
                    k_stmt(
                        #k_place_holder()
                        #k_action('turnLeft')
                        k_action('putMarker')
                    ),
                    k_stmt(
                        k_place_holder()
                    )
                )
            )
        )
    print(p)

    # karel states logger
    L = KarelStatesLogger()

    # whole program execution
    K.draw(with_color=True)
    karel_state = K.execute(program=p, states_logger=L, max_steps=None)
    K.draw(with_color=True)


    # single stmt program execution
    K.execute_single_stmt(stmt=k_stmt(k_action('turnLeft')))
    K.draw(with_color=True)


    # single stmt program execution
    new_stmt = k_action('move')
    K.execute_single_stmt(new_stmt)
    K.draw(with_color=True)


    # locate the place holder hook
    PHH = p.register()
    print(type(PHH))  # <class stmt>

    # update the place holder
    PHH.function = k_action('move')
    print(p)


##############
# Parallel Test
##############
def exe_test(seed):

    # executor
    E = KarelRobotRandomMaze(batch_size=10, seed=seed)
    #E = KarelRobotFourCorners(batch_size=10, seed=seed)
    #E = KarelRobotStairClimer(batch_size=10, seed=seed)
    #E = KarelRobotHarvester(batch_size=10, seed=seed)
    #E = KarelRobotCleanHouse(batch_size=10, seed=seed)
    #E = KarelRobotTopOff(batch_size=10, seed=seed)

    print('[start program]', E.program)

    done = False
    mask = None
    num_predictions = 0
    max_predictions = 10
    while not done and num_predictions <= max_predictions:
        code, mask, done = E.predict(mask=mask)
        if not done:
            E.mount(code)
        num_predictions += 1
    
    print('[predicted program]', E.program)
    
    # find the completed while loop?
    exit()


    # execute and compute the reward
    E.execute(max_steps=100)

    R = E.compute_reward()
    print(R)


if __name__ == "__main__":

    import sys

    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 123

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    #simple_test()
    exe_test(SEED)