import pdb

import numpy as np
from tqdm import tqdm

import minigrid_base_dsl
from utils.convert_prog import ConvertProg
from utils.logging import log_and_print


# randomCrossing
def get_dsl_dict():
    dsl_dict = {
        # flow control
        "IF": minigrid_base_dsl.IF,
        "WHILE": minigrid_base_dsl.WHILE,
        "IFELSE": minigrid_base_dsl.IFELSE,
        "END": minigrid_base_dsl.END,
    }

    # add predicate
    for key, val in minigrid_base_dsl.COND_DICT.items():
        dsl_dict[key] = val

    # add actions
    for key, val in minigrid_base_dsl.ACTION_DICT.items():
        dsl_dict[key] = minigrid_base_dsl.ACTION(val)

    return dsl_dict


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
