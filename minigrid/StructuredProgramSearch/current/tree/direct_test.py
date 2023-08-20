# NOTE: node in the tree, contains a sketch and a queue for searching
import pdb

from dsl import *

from karel.robot import KarelRobot
from utils.logging import init_logging

if __name__ == "__main__":
    task='topOffPick'
    seed = 888

    init_logging('store/robot_trace', 'log_{}_{}.txt'.format(task, seed))

    # test first
    # if_1 = IF(COND_DICT['markers_present'])
    # if_1.stmts = [ACTION(ACTION_DICT['pick_marker']), ACTION(ACTION_DICT['pick_marker'])]
    # if_2 = IF(COND_DICT['right_is_clear'])
    # if_2.stmts = [ACTION(ACTION_DICT['turn_right']), ACTION(ACTION_DICT['move'])]
    # a_1 = ACTION(ACTION_DICT['move'])
    # a_2 = ACTION(ACTION_DICT['move'])
    # a_3 = ACTION(ACTION_DICT['pick_marker'])
    # a_4 = ACTION(ACTION_DICT['turn_left'])
    # while_1 = WHILE()
    # while_1.cond = [COND_DICT['markers_present']]
    # while_1.stmts = [a_3, if_1, a_1, a_2]
    # while_2 = WHILE()
    # while_2.cond = [COND_DICT['front_is_clear']]
    # while_2.stmts = [while_1, if_2, a_4]
    # program = Program()
    # program.stmts = [while_2, program.stmts[-1]]

    if_1 = IF(COND_DICT['not(front_is_clear)'])
    if_1.stmts = [ACTION(ACTION_DICT['turn_left'])]
    a_1 = ACTION(ACTION_DICT['pick_marker'])
    a_2 = ACTION(ACTION_DICT['move'])
    while_1 = WHILE()
    while_1.cond = [COND_DICT['not(right_is_clear)']]
    while_1.stmts = [if_1, a_1, a_2]
    program = Program()
    program.stmts = [while_1, program.stmts[-1]]

    # robot
    print(program)
    pdb.set_trace()
    # more_seeds = [0, 999, 123, 666, 546, 11, 4372185, 6431, 888, 1, 2, 3, 4, 5, 0]
    more_seeds = [seed]
    rewards = []
    for seed in more_seeds:
        force_eval_robot = KarelRobot(task='topOffPick', seed=seed)
        force_eval_robot.draw()
        force_eval_robot.force_execution = True
        program.execute(force_eval_robot)
        program.reset()
        r = force_eval_robot.check_reward()
        rewards.append(r)
    
    # pdb.set_trace()
    print(rewards)