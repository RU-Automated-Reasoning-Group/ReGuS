import random

import numpy as np
from dsl import *

from karel.robot import KarelRobot

if __name__ == "__main__":

    random.seed(123)
    np.random.seed(123)

    robot = KarelRobot(task='randomMaze', seed=0)
    robot.draw()
    exit()

    for i in range(10000):
        robot = KarelRobot(task='topOff', seed=i)
        if [10, 7] in robot.checker.markers_pos and [10, 6] in robot.checker.markers_pos:
            robot.draw()
            print('[seed]', i)
            print('[num markers]', robot.checker.num_markers)
            exit()
    exit()

    robot = KarelRobot(task='randomMaze', seed=123)
    robot.draw()

    robot = KarelRobot(task='randomMaze', seed=666)
    robot.draw()

    robot = KarelRobot(task='randomMaze', seed=321)
    robot.draw()

    robot = KarelRobot(task='randomMaze', seed=5432)  # identical?
    robot.draw()

    robot = KarelRobot(task='randomMaze', seed=65246)
    robot.draw()