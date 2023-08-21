import pdb

import numpy as np
from tqdm import tqdm

# from robot_dsl import *
from minigrid_base_dsl import *
# from minigrid_lava_dsl import *
# from minigrid_multiroom_dsl import *
# from minigrid_lockedroom_dsl import *
# from minigrid_unlockpickup_dsl import *
from utils.convert_prog import ConvertProg
from utils.logging import log_and_print

# def get_dsl_dict():
#     dsl_dict = {
#         # flow control
#         'IF': IF,
#         'WHILE': WHILE,
#         'IFELSE': IFELSE,
#         'END': END(),
#         # predicate
#         'block_at_goal': COND_DICT['block_at_goal'],
#         'block_is_grasped': COND_DICT['block_is_grasped'],
#         'block_above_goal': COND_DICT['block_above_goal'],
#         'block_below_gripper': COND_DICT['block_below_gripper'],
#         'block_inside_gripper': COND_DICT['block_inside_gripper'],
#         'gripper_are_open': COND_DICT['gripper_are_open'],

#         'not(block_at_goal)': COND_DICT['not(block_at_goal)'],
#         'not(block_is_grasped)': COND_DICT['not(block_is_grasped)'],
#         'not(block_above_goal)': COND_DICT['not(block_above_goal)'],
#         'not(block_below_gripper)': COND_DICT['not(block_below_gripper)'],
#         'not(block_inside_gripper)': COND_DICT['not(block_inside_gripper)'],
#         'not(gripper_are_open)': COND_DICT['not(gripper_are_open)'],
#         # actions
#         'move_to_block': ACTION(ACTION_DICT['move_to_block']),
#         'move_to_goal': ACTION(ACTION_DICT['move_to_goal']),
#         'idle': ACTION(ACTION_DICT['idle']),
#         'move_down': ACTION(ACTION_DICT['move_down']),
#         'open_gripper': ACTION(ACTION_DICT['open_gripper']),
#         'close_gripper': ACTION(ACTION_DICT['close_gripper'])
#     }

#     return dsl_dict


# randomCrossing
def get_dsl_dict():
    dsl_dict = {
        # flow control
        "IF": IF,
        "WHILE": WHILE,
        "IFELSE": IFELSE,
        "END": END(),
        # predicate
        "front_is_clear": COND_DICT["front_is_clear"],
        "left_is_clear": COND_DICT["left_is_clear"],
        "right_is_clear": COND_DICT["right_is_clear"],
        "goal_on_left": COND_DICT["goal_on_left"],
        "goal_on_right": COND_DICT["goal_on_right"],
        "goal_present": COND_DICT["goal_present"],
        "not(front_is_clear)": COND_DICT["not(front_is_clear)"],
        "not(left_is_clear)": COND_DICT["not(left_is_clear)"],
        "not(right_is_clear)": COND_DICT["not(right_is_clear)"],
        "not(goal_on_left)": COND_DICT["not(goal_on_left)"],
        "not(goal_on_right)": COND_DICT["not(goal_on_right)"],
        "not(goal_present)": COND_DICT["not(goal_present)"],
        # actions
        "move": ACTION(ACTION_DICT["move"]),
        "turn_left": ACTION(ACTION_DICT["turn_left"]),
        "turn_right": ACTION(ACTION_DICT["turn_right"]),
        "return": ACTION(ACTION_DICT["return"]),
    }
    return dsl_dict


# # lava crossing
# def get_dsl_dict():
#     dsl_dict = {
#         # flow control
#         "IF": IF,
#         "WHILE": WHILE,
#         "IFELSE": IFELSE,
#         "END": END(),
#         # predicate
#         "front_is_clear": COND_DICT["front_is_clear"],
#         "left_is_clear": COND_DICT["left_is_clear"],
#         "right_is_clear": COND_DICT["right_is_clear"],
#         "goal_on_left": COND_DICT["goal_on_left"],
#         "goal_on_right": COND_DICT["goal_on_right"],
#         "goal_present": COND_DICT["goal_present"],
#         "front_is_lava": COND_DICT["front_is_lava"],
#         "not(front_is_clear)": COND_DICT["not(front_is_clear)"],
#         "not(left_is_clear)": COND_DICT["not(left_is_clear)"],
#         "not(right_is_clear)": COND_DICT["not(right_is_clear)"],
#         "not(goal_on_left)": COND_DICT["not(goal_on_left)"],
#         "not(goal_on_right)": COND_DICT["not(goal_on_right)"],
#         "not(goal_present)": COND_DICT["not(goal_present)"],
#         "not(front_is_lava)": COND_DICT["not(front_is_lava)"],
#         # actions
#         "move": ACTION(ACTION_DICT["move"]),
#         "turn_left": ACTION(ACTION_DICT["turn_left"]),
#         "turn_right": ACTION(ACTION_DICT["turn_right"]),
#         "return": ACTION(ACTION_DICT["return"]),
#     }
#     return dsl_dict


# def get_dsl_dict():
#     dsl_dict = {
#         # flow control
#         "IF": IF,
#         "WHILE": WHILE,
#         "IFELSE": IFELSE,
#         "END": END(),
#         # predicate
#         "front_is_clear": COND_DICT["front_is_clear"],
#         "left_is_clear": COND_DICT["left_is_clear"],
#         "right_is_clear": COND_DICT["right_is_clear"],
#         "goal_on_left": COND_DICT["goal_on_left"],
#         "goal_on_right": COND_DICT["goal_on_right"],
#         "goal_present": COND_DICT["goal_present"],
#         "not(front_is_clear)": COND_DICT["not(front_is_clear)"],
#         "not(left_is_clear)": COND_DICT["not(left_is_clear)"],
#         "not(right_is_clear)": COND_DICT["not(right_is_clear)"],
#         "not(goal_on_left)": COND_DICT["not(goal_on_left)"],
#         "not(goal_on_right)": COND_DICT["not(goal_on_right)"],
#         "not(goal_present)": COND_DICT["not(goal_present)"],
#         # actions
#         "move": ACTION(ACTION_DICT["move"]),
#         "turn_left": ACTION(ACTION_DICT["turn_left"]),
#         "turn_right": ACTION(ACTION_DICT["turn_right"]),
#         "toggle": ACTION(ACTION_DICT["toggle"]),
#         "RC_get": ACTION(ACTION_DICT["RC_get"]),
#     }
#     return dsl_dict


# locked room
# def get_dsl_dict():
#     dsl_dict = {
#         # flow control
#         "IF": IF,
#         "WHILE": WHILE,
#         "IFELSE": IFELSE,
#         "END": END(),
#         # predicate
#         "front_is_clear": COND_DICT["front_is_clear"],
#         "left_is_clear": COND_DICT["left_is_clear"],
#         "right_is_clear": COND_DICT["right_is_clear"],
#         "goal_on_left": COND_DICT["goal_on_left"],
#         "goal_on_right": COND_DICT["goal_on_right"],
#         "goal_present": COND_DICT["goal_present"],
#         "front_is_closed_door": COND_DICT["front_is_closed_door"],
#         "front_is_locked_door": COND_DICT["front_is_locked_door"],
#         "front_is_wall": COND_DICT["front_is_wall"],
#         "front_is_obj": COND_DICT["front_is_obj"],
#         "front_is_key": COND_DICT["front_is_key"],
#         "not(front_is_clear)": COND_DICT["not(front_is_clear)"],
#         "not(left_is_clear)": COND_DICT["not(left_is_clear)"],
#         "not(right_is_clear)": COND_DICT["not(right_is_clear)"],
#         "not(goal_on_left)": COND_DICT["not(goal_on_left)"],
#         "not(goal_on_right)": COND_DICT["not(goal_on_right)"],
#         "not(goal_present)": COND_DICT["not(goal_present)"],
#         "not(front_is_closed_door)": COND_DICT["not(front_is_closed_door)"],
#         "not(front_is_locked_door)": COND_DICT["not(front_is_locked_door)"],
#         "not(front_is_wall)": COND_DICT["not(front_is_wall)"],
#         "not(front_is_obj)": COND_DICT["not(front_is_obj)"],
#         "not(front_is_key)": COND_DICT["not(front_is_key)"],
#         # actions
#         "move": ACTION(ACTION_DICT["move"]),
#         "turn_left": ACTION(ACTION_DICT["turn_left"]),
#         "turn_right": ACTION(ACTION_DICT["turn_right"]),
#         "return": ACTION(ACTION_DICT["return"]),
#         "toggle": ACTION(ACTION_DICT["toggle"]),
#         "pickup": ACTION(ACTION_DICT["pickup"]),
#         "get_key": ACTION(ACTION_DICT["get_key"]),
#         "get_goal": ACTION(ACTION_DICT["get_goal"]),
#         "get_locked_door": ACTION(ACTION_DICT["get_locked_door"]),
#     }
#     return dsl_dict


# unlock pickup
# def get_dsl_dict():
#     dsl_dict = {
#         # flow control
#         "IF": IF,
#         "WHILE": WHILE,
#         "IFELSE": IFELSE,
#         "END": END(),
#         # predicate
#         "front_is_clear": COND_DICT["front_is_clear"],
#         "left_is_clear": COND_DICT["left_is_clear"],
#         "right_is_clear": COND_DICT["right_is_clear"],
#         "clear_to_drop": COND_DICT["clear_to_drop"],
#         "has_key": COND_DICT["has_key"],
#         "not(front_is_clear)": COND_DICT["not(front_is_clear)"],
#         "not(left_is_clear)": COND_DICT["not(left_is_clear)"],
#         "not(right_is_clear)": COND_DICT["not(right_is_clear)"],
#         "not(clear_to_drop)": COND_DICT["not(clear_to_drop)"],
#         "not(has_key)": COND_DICT["not(has_key)"],
#         # actions
#         "move": ACTION(ACTION_DICT["move"]),
#         "turn_left": ACTION(ACTION_DICT["turn_left"]),
#         "turn_right": ACTION(ACTION_DICT["turn_right"]),
#         "return": ACTION(ACTION_DICT["return"]),
#         "pickup": ACTION(ACTION_DICT["pickup"]),
#         "drop": ACTION(ACTION_DICT["drop"]),
#         "get_box": ACTION(ACTION_DICT["get_box"]),
#         "get_goal": ACTION(ACTION_DICT["get_goal"]),
#     }
#     return dsl_dict


# doorkey
# def get_dsl_dict():
#     dsl_dict = {
#         # flow control
#         "IF": IF,
#         "WHILE": WHILE,
#         "IFELSE": IFELSE,
#         "END": END(),
#         # predicate
#         "front_is_clear": COND_DICT["front_is_clear"],
#         "left_is_clear": COND_DICT["left_is_clear"],
#         "right_is_clear": COND_DICT["right_is_clear"],
#         "goal_on_left": COND_DICT["goal_on_left"],
#         "goal_on_right": COND_DICT["goal_on_right"],
#         "goal_present": COND_DICT["goal_present"],
#         "front_is_closed_door": COND_DICT["front_is_closed_door"],
#         "front_is_locked_door": COND_DICT["front_is_locked_door"],
#         "not(front_is_clear)": COND_DICT["not(front_is_clear)"],
#         "not(left_is_clear)": COND_DICT["not(left_is_clear)"],
#         "not(right_is_clear)": COND_DICT["not(right_is_clear)"],
#         "not(goal_on_left)": COND_DICT["not(goal_on_left)"],
#         "not(goal_on_right)": COND_DICT["not(goal_on_right)"],
#         "not(goal_present)": COND_DICT["not(goal_present)"],
#         "not(front_is_closed_door)": COND_DICT["not(front_is_closed_door)"],
#         "not(front_is_locked_door)": COND_DICT["not(front_is_locked_door)"],
#         # actions
#         "move": ACTION(ACTION_DICT["move"]),
#         "turn_left": ACTION(ACTION_DICT["turn_left"]),
#         "turn_right": ACTION(ACTION_DICT["turn_right"]),
#         "return": ACTION(ACTION_DICT["return"]),
#         "toggle": ACTION(ACTION_DICT["toggle"]),
#         "pickup": ACTION(ACTION_DICT["pickup"]),
#         "get_key": ACTION(ACTION_DICT["get_key"]),
#         "get_locked_door": ACTION(ACTION_DICT["get_locked_door"]),
#     }
#     return dsl_dict


def test():
    dsl_dict = get_dsl_dict()
    convertor = ConvertProg(dsl_dict)

    df = pd.read_csv("store/highway_log/keep/no_fuel_new_2.csv", header=None)
    progs = df[1]

    c_prob = 0
    for test_prog in tqdm(progs):
        if "C" in test_prog:
            c_prob += 1
            print("C exist for: {}".format(test_prog))
            continue

        prog_nice = test_prog.replace("not ", "not")
        get_prog = convertor.get_prog(prog_nice)
        if str(get_prog).strip() != prog_nice:
            pdb.set_trace()

    print("c problem: {}".format(c_prob))


def do_test(program):
    eval_seeds = [4000 * i + 5000 for i in range(10)]
    rewards = []

    for seed in tqdm(eval_seeds):
        force_eval_robot = HighwayRobot(seed=seed)
        force_eval_robot.max_steps = 120
        force_eval_robot.force_execution = True
        program.execute(force_eval_robot)
        r = force_eval_robot.check_reward()
        rewards.append(r)
        program.reset()
        if r == -1:
            break

    return rewards


def test_highway_test():
    dsl_dict = get_dsl_dict()
    convertor = ConvertProg(dsl_dict)

    df = pd.read_csv("store/highway_log/keep/no_fuel_new.csv", header=None)
    progs = df[1]

    for test_prog in tqdm(progs):
        if "C" in test_prog:
            continue

        prog_nice = test_prog.replace("not ", "not")
        get_prog = convertor.get_prog(prog_nice)
        if str(get_prog).strip() != prog_nice:
            pdb.set_trace()

        log_and_print("current prog: {}".format(str(get_prog)))
        rewards = do_test(get_prog)
        if -1 in rewards:
            log_and_print("fail")
        else:
            log_and_print(np.mean(rewards))
        log_and_print(" ")

    return rewards


def display_prog():
    dsl_dict = get_dsl_dict()
    ConvertProg(dsl_dict)

    df = pd.read_csv("store/highway_log/keep/no_fuel_new.csv", header=None)
    progs = df[1]

    for test_prog in tqdm(progs):
        if "lane_right" in test_prog and "faster" in test_prog:
            print(test_prog)
            pdb.set_trace()


def convert_program(prog_str):
    dsl_dict = get_dsl_dict()
    converter = ConvertProg(dsl_dict)

    prog_nice = prog_str.replace("not ", "not")
    return converter.get_prog(prog_nice)


if __name__ == "__main__":
    # init_logging('store/highway_log', 'test_new_2.txt')
    # test_highway_test()

    p = convert_program(
        "WHILE(not (block_at_goal)) { IF(gripper_are_open) { IF(not (block_inside_gripper)) { move_down} } IF(block_inside_gripper) { IF(block_is_grasped) { move_to_goal} ELSE { close_gripper close_gripper}} ELSE { open_gripper} idle} "
    )
    print("end")
