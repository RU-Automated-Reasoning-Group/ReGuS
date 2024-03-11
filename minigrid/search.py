import argparse
import copy

# NOTE: each node is associated with a sketch
import os
import pickle
import queue
import random
import time
import multiprocessing as mp
from queue import PriorityQueue

import numpy as np

# for minigrid environment
from minigrid_implement.robot import MiniGridRobot
import minigrid_implement.dsl

import minigrid_base_dsl
Robot = MiniGridRobot

import pdb
import sys
import debug_program
from utils.logging import log_and_print

import minigrid
minigrid.register_minigrid_envs()

class SUCC(Exception):
    def __init__(self, p):
        self.p = copy.deepcopy(p)

def execute_for_reward(cur_robot, program, force=True):
    cur_prog = copy.deepcopy(program)
    cur_prog.reset_resume_points()
    cur_prog.reset()
    cur_robot.force_execution = force

    cur_prog.execute(cur_robot)

    return cur_robot.no_fuel(), cur_robot.check_reward(), cur_prog.complete()


def save_prog(program, filename):
    with open(filename, "wb") as f:
        pickle.dump(program, f)
    print(f"program saved to {filename}")


def initialize_robot(task, n):
    for i in range(0, n):
        r = Robot(task, seed=i)
        r.render(f"frames/{i}.png")
        print(f"created robot {i}")


def copy_robot(robot):
    robot_copy = copy.deepcopy(robot)
    robot_copy.env.sim.set_state(copy.deepcopy(robot.env.sim.get_state()))
    robot_copy.env.sim.forward()
    robot_copy.env.env.goal = copy.deepcopy(robot.env.goal)
    robot_copy.env.env.initial_state = copy.deepcopy(robot.env.initial_state)
    robot_copy.env.env.obs = copy.deepcopy(robot.env.obs)
    # robot_copy.env.reward_info = 'place'
    return robot_copy


def get_structural_cost(program):
    cost = 0
    program_str = str(program)
    for s in minigrid_base_dsl.ACTION_DICT:
        cost += program_str.count(s)
    cost -= program_str.count("clear_to_drop")
    return cost


def execute_single_action_with_library(action, robot):
    # action is a action object
    if action.action.action in minigrid_base_dsl.LIBRARY_DICT:
        action = minigrid_base_dsl.get_action(action, None, None)
        action.execute(robot, None)
        robot.active = True
        robot.returned = False
    else:
        robot.execute_single_action(action.action)


# NOTE: each node is associated with a sketch
class Node:
    # constant
    SUCCESS_TYPE = "success"
    FAIL_TYPE = "fail"
    MORE_WORK_TYPE = "more_work"

    # init
    def __init__(
        self,
        sketch,
        task,
        seed=123,
        more_seeds=[],
        eval_seeds=[],
        max_search_iter=50,
        max_structural_cost=30,
        shuffle_actions=False,
        found_one=False,
        make_video=False,
    ):
        self.sketch = sketch
        self.task = task
        self.seed = seed
        self.more_seeds = more_seeds
        self.eval_seeds = eval_seeds
        self.max_search_iter = max_search_iter
        self.max_structural_cost = max_structural_cost
        self.shuffle_actions = shuffle_actions
        self.found_one = found_one
        self.make_video = make_video

        # store all required robot
        self.robot_store = {self.seed: Robot(self.task, seed=self.seed)}
        # add environment specific actions here
        # robot = MiniGridRobot(self.task, seed=self.seed)
        # mission_str = robot.env.env.mi
        for e in more_seeds:
            self.robot_store[e] = Robot(self.task, seed=e, make_video=self.make_video)
        for e in eval_seeds:
            self.robot_store[e] = Robot(self.task, seed=e, make_video=self.make_video)
        assert len(eval_seeds) > 0

        self.q = PriorityQueue()
        self.q.put(
            (
                0,  # reward
                time.time(),  # timestamp
                copy.deepcopy(self.sketch),
                self.get_robot(self.seed),
                None,  # last evaluation
                None,  # cost
            )
        )
        # NOTE: tmp code, store potential case 3 programs? not so sure
        self.case_3_list = []

        self.candidates = {  # store success (and unsuccess) programs
            "success": [],
            "success_search": [],
            "failed": [],
            "no_fuel": [],
            "complete": [],
        }

        # TODO: # number of top-k programs we select from a node
        self.k = 1

        # multiprocess
        self.pool = mp.Pool(processes=len(self.more_seeds) + 1)

    def clean(self):
        del self.pool
        del self.q
    # get robot from store (used for multiple simulator)
    # def get_robot(self, seed):
    #     robot = self.robot_store[seed]
    #     robot_copy = copy.deepcopy(robot)
    #     robot_copy.env.sim.set_state(copy.deepcopy(robot.env.sim.get_state()))
    #     robot_copy.env.sim.forward()
    #     robot_copy.env.env.goal = copy.deepcopy(robot.env.goal)
    #     robot_copy.env.env.initial_state = copy.deepcopy(robot.env.initial_state)
    #     robot_copy.env.env.obs = copy.deepcopy(robot.env.obs)
    #     # robot_copy.env.reset()
    #     # robot_copy.env.reward_info = 'place'
    #     return robot_copy
    def execute_for_reward(self, cur_robot, program, force=True):
        cur_prog = copy.deepcopy(program)
        cur_prog.reset_resume_points()
        cur_prog.reset()
        cur_robot.force_execution = force

        cur_prog.execute(cur_robot)

        return cur_robot.no_fuel(), cur_robot.check_reward(), cur_prog.complete()

    def remove_and_update(self, program):
        program.remove_abs_state()
        for e in self.eval_seeds:
            robot = self.get_robot(e)
            program.execute_and_update(robot)
            if robot.check_reward() == 1:
                done = "success"
            else:
                done = "fail"
            print(f"seed {e} completed ({done})")
        print(program.to_string_verbose())
        return program
    
    def execute_and_update(self, program):
        for e in self.eval_seeds:
            robot = self.get_robot(e)
            program.execute_and_update(robot)
            if robot.check_reward() == 1:
                done = "success"
            else:
                done = "fail"
            print(f"seed {e} completed ({done})")
        print(program.to_string_verbose())
        return program

    # used for single simulator
    def get_robot(self, seed):
        robot = Robot(self.task, seed, make_video=self.make_video)
        # robot.force_execution = True
        # ACTION(k_action("get_box")).execute(robot, None)
        # robot.force_execution = False

        return robot

    def copy_robot(self, robot):
        robot_copy = copy.deepcopy(robot)
        return robot_copy

    # add candidate program into queue
    def add_queue(self, candidate, reward, robot, cost=None, is_return=False):
        # calculate reward before add
        print("trying to add", candidate)
        eval_result, robot, reward = self.eval_program(
            self.seed, robot, candidate, check_multiple=False
        )
        # reward = robot.check_reward()
        if is_return and reward != 1:
            return cost, False
        # add back

        # cover the case that the last while loop is only used once
        if (
            candidate.count_C() <= 2
            and type(candidate.stmts[-2]) == minigrid_base_dsl.C
            and robot.reward == 1
        ):
            reward = 1
            eval_result = self.SUCCESS_TYPE

        if reward == 1:
            # now just evaluate in one thread due to the implementation of the fetch environment
            # result = []
            # for e in self.more_seeds:
            #     r = execute_for_reward(self.get_robot(e), candidate, Fa)
            # results = [
            #     self.pool.apply_async(
            #         execute_for_reward, args=(self.get_robot(e), candidate, True)
            #     )
            #     for e in self.more_seeds
            # ]
            
            # # for e in self.more_seeds:
            # #     execute_for_reward(self.get_robot(e), candidate, True)

            # results = [p.get() for p in results]
            # all_rewards = [x[1] for x in results]
            # reward = np.mean(all_rewards)
            # for idx, s in enumerate(self.more_seeds):
            #     print(f"seed {s} with reward {all_rewards[idx]}")
            # print(f"reward updated from 1 to {reward}")

            # todo:
            if reward == 1 and candidate.find_actions()[0] is None:
                print("evaulating program")
                success_rate = self.test_for_success_rate(candidate)
                if success_rate == 1.0:
                    print("found a correct program")
                    raise SUCC(candidate)

        # assert cost is None
        if cost is None:
            cost = get_structural_cost(candidate)
        q_tuple = (
            # -reward + cost * 0.02,
            # -reward + cost,
            -4 * reward + cost * 0.5,
            # -reward + cost * 0.1,
            # -reward + cost * 0.2,
            time.time(),
            candidate,
            robot,
            (eval_result, reward),
            None,
        )
        if not cost > self.max_structural_cost:
            self.q.put(q_tuple)
            print(f"equeued reward {reward} cost {cost} final {q_tuple[0]}")
            return cost, True
        return cost, False

    # check whether C exist in statement
    def check_C_exist(self, stmts):
        exist = False
        for idx, code in enumerate(stmts):
            if isinstance(
                code,
                (
                    minigrid_base_dsl.S,
                    minigrid_base_dsl.B,
                    minigrid_base_dsl.ACTION,
                    minigrid_base_dsl.HIDE_ACTION,
                    minigrid_base_dsl.k_cond,
                    minigrid_base_dsl.END,
                ),
            ):
                pass
            elif isinstance(
                code,
                (
                    minigrid_base_dsl.WHILE,
                    minigrid_base_dsl.IF,
                    minigrid_base_dsl.IFELSE,
                ),
            ):
                exist = self.check_C_exist(code.stmts)
                if exist:
                    return exist
            elif isinstance(code, minigrid_base_dsl.C):
                return True
            else:
                raise ValueError("Invalide code")

        return exist

    def test_program(self, prog):
        minigrid_implement.dsl.SEARCH_STATUS = False
        for e in self.eval_seeds:
            # force evaluate
            force_eval_robot = self.get_robot(e)
            force_eval_robot.force_execution = True
            prog.execute(force_eval_robot)
            prog.reset()
            force_eval_robot.force_execution = False
            passed = (
                force_eval_robot.env.env.get_reward() == 1
                and not force_eval_robot.no_fuel()
            )
            if passed:
                print(f"seed {e} passed", flush=True)
            else:
                print(f"seed {e} failed", flush=True)
            # fail
            if not passed:
                # log and print
                log_and_print(
                    "\nfound but not success in all seeds for \n {}".format(prog)
                )
                # print("successful rate", self.test_for_success_rate(prog))
                print("location 1")
                print(f"seed {str(e)} failed")
                self.candidates["success_search"].append((1, prog))
                minigrid_implement.dsl.SEARCH_STATUS = True
                return self.FAIL_TYPE
        # success
        log_and_print("\nsuccess and store for \n {}".format(prog))
        # if self.task == "MiniGrid-RandomCrossingS11N5-v0":
        #     # pdb.set_trace()
        #     if (
        #         str(prog)
        #         == " WHILE(not (goal_on_right)) { IF(left_is_clear) { turn_left}  IF(goal_present) { return}  IF(not (front_is_clear)) { turn_right}  move} ; turn_right WHILE(not (goal_present)) { move} ;; END"
        #     ):
        #         minigrid_implement.dsl.print_interaction()
        #         exit()
        # elif self.task == "MiniGrid-RandomLavaCrossingS11N5-v0":
        #     # if (
        #     # str(prog)
        #     # == " WHILE(not (goal_on_right)) { IF(left_is_clear) { turn_left}  IF(goal_present) { return}  IF(not (front_is_clear)) { turn_right}  IF(front_is_lava) { turn_right}  move} ; turn_right WHILE(not (goal_present)) { move} ;; END"
        #     # ):
        #     if True:
        #         minigrid_implement.dsl.print_interaction()
        #         exit()
        # elif self.task == "MiniGrid-MultiRoomNoDoor-N6-v0":
        #     if (
        #         str(prog)
        #         == " WHILE(not (front_is_wall)) { move} ; turn_right RC_get; END"
        #     ):
        #         minigrid_implement.dsl.print_interaction()
        #         exit()
        # elif self.task == "MiniGrid-MultiRoom-N6-v0":
        #     if (
        #         str(prog)
        #         == " WHILE(not (front_is_wall)) { IF(front_is_obj) { turn_right}  move} ; turn_right WHILE(not (goal_on_right)) { IF(left_is_clear) { turn_left}  IF(goal_present) { return}  IF(front_is_closed_door) { toggle}  IF(not (front_is_clear)) { turn_right}  move} ; turn_right WHILE(not (goal_present)) { move} ;; END"
        #     ):
        #         minigrid_implement.dsl.print_interaction()
        #         exit()
        # elif self.task == "MiniGrid-LockedRoom-v0":
        #     if (
        #         str(prog)
        #         == " WHILE(not (front_is_wall)) { IF(front_is_obj) { turn_right}  move} ; turn_right WHILE(not (goal_on_right)) { IF(left_is_clear) { turn_left}  IF(goal_present) { return}  IF(front_is_closed_door) { IF(front_is_locked_door) { get_key pickup get_locked_door}  toggle}  IF(not (front_is_clear)) { turn_right}  move} ; turn_right WHILE(not (goal_present)) { move} ;; END"
        #     ):
        #         minigrid_implement.dsl.print_interaction()
        #         exit()
        # elif self.task == "MiniGrid-DoorKey-8x8-v0":
        #     # if str(prog) == " WHILE(not (front_is_wall)) { IF(front_is_obj) { turn_right}  move} ; turn_right WHILE(not (goal_on_right)) { IF(front_is_key) { pickup}  IF(left_is_clear) { turn_left}  IF(goal_present) { return}  IF(front_is_closed_door) { IF(front_is_locked_door) { IF(not (has_key)) { get_key pickup}  get_locked_door}  toggle}  IF(not (front_is_clear)) { IF(not (front_is_key)) { turn_right} }  move} ; turn_right WHILE(not (goal_present)) { move} ;; END":
        #     minigrid_implement.dsl.print_interaction()
        #     exit()
        self.candidates["success"].append((1, prog))
        minigrid_implement.dsl.SEARCH_STATUS = True
        return self.SUCCESS_TYPE

    def test_for_success_rate(self, prog):
        count = 0
        # lst = [self.seed]
        lst = self.eval_seeds
        failed = []
        for e in lst:
            print(f"============== seed {e} ==============")
            # force evaluate
            force_eval_robot = Robot(self.task, e)
            force_eval_robot.force_execution = True
            prog.execute(force_eval_robot)
            prog.reset()
            force_eval_robot.force_execution = False
            # if force_eval_robot.check_success():
            if force_eval_robot.env.env.get_reward() == 1:
                # pdb.set_trace()
                count += 1
                print(f"seed {e} success")
            else:
                # pdb.set_trace()
                print(f"seed {e} fail")
                failed.append(e)
        # print(f"count: {count}, len {len(self.eval_seeds)}")
        print(failed)
        self.more_seeds.extend(failed)
        return count / len(lst)

    def test_success_rate_env(self, num_of_seeds, env_name, prog):
        count = 0
        for i in range(0, num_of_seeds):
            force_eval_robot = Robot(env_name, i, 2000)
            force_eval_robot.env.env.max_steps = 2000
            force_eval_robot.force_execution = True
            prog.reset()
            prog.reset_resume_points()
            prog.execute(force_eval_robot)
            prog.reset_resume_points()
            prog.reset()
            if force_eval_robot.check_reward() == 1:
                count += 1
            else:
                print(f"seed {i} fail")
        print(f"count: {count}, len {num_of_seeds}")
        return count / num_of_seeds

    def eval_program(self, seed, robot, candidate, check_multiple):
        always_multi_seed_evaluation = True
        single_seed = len(self.more_seeds) == 0
        check_multiple = False if single_seed else check_multiple

        eval_robot = robot
        # pdb.set_trace()
        candidate.execute(eval_robot)
        r = eval_robot.check_reward()
        complete = candidate.complete()
        seed = eval_robot.seed

        # fail
        if r == -1:
            # log and print
            log_and_print("\n fail for \n {}".format(candidate))
            self.candidates["failed"].append((-1, candidate))
            return self.FAIL_TYPE, eval_robot, -1

        # success or more work
        all_rewards = [r]
        all_no_fuel = [eval_robot.no_fuel()]
        all_completes = [complete]
        # success_seeds = [seed]
        all_seeds = [self.seed] + self.more_seeds
        all_seeds.pop(all_seeds.index(seed))

        # results = [
        #         self.pool.apply_async(
        #             execute_for_reward, args=(self.get_robot(e), candidate, True)
        #         )
        #         for e in self.more_seeds
        #     ]
        # results = [p.get() for p in results]
        results = []
        for e in self.more_seeds:
            results.append(execute_for_reward(self.get_robot(e), candidate, True))

        # pdb.set_trace() 
        for tmp_seed, (robot_no_fuel, reward, complete) in zip(self.more_seeds, results):
            # C or breakpoint
            all_rewards.append(reward)
            all_no_fuel.append(robot_no_fuel)
            all_completes.append(complete)
            print(f"{tmp_seed} with reward {reward}")

        # check fail
        all_seeds = [seed] + all_seeds
        for seed, robot_no_fuel, reward, complete in zip(all_seeds, all_no_fuel, all_rewards, all_completes):
            if reward == -1:
                # log and print
                log_and_print('\n fail for \n {}'.format(candidate))
                self.candidates['failed'].append((-1, candidate))
                return self.FAIL_TYPE, eval_robot, -1
            elif reward < 1.0 and robot_no_fuel:
                # log and print
                log_and_print('\n no fuel with reward {} under seed {} for \n {}'.format(reward, seed, candidate))
                self.candidates['no_fuel'].append((r, candidate))
                return self.FAIL_TYPE, eval_robot, -1
            elif reward < 1.0 and complete:
                # log and print
                log_and_print('\n complete with reward {} under seed {} for\n {}'.format(reward, seed, candidate))
                self.candidates['complete'].append((r, candidate))
                return self.FAIL_TYPE, eval_robot, -1

        # more work on search seed
        if r < 1.0:
            if candidate.count_C() == 1 and isinstance(candidate.stmts[-2], minigrid_base_dsl.C):
                log_and_print('\n special complete with reward {} for\n {}'.format(np.mean(all_rewards), candidate))
                self.candidates['complete'].append((np.mean(all_rewards), candidate))
            
            return self.MORE_WORK_TYPE, eval_robot, np.mean(all_rewards)

        # check success
        if np.mean(np.array(all_rewards) >= 1.0) == 1:
            # log and print
            log_and_print('\n success and store for {}'.format(candidate))
            self.candidates['success'].append((1.0, candidate))
            return self.SUCCESS_TYPE, eval_robot, 1.0
        else:
            log_and_print('\nfound but not success in all seeds with reward {} for \n {}'.format(np.mean(all_rewards), candidate))
            self.candidates['success_search'].append((np.mean(all_rewards), candidate))
            
            for seed, reward in zip(all_seeds, all_rewards):
                if reward < 1.0:
                    new_seed = seed
                    break

            candidate.reset()
            eval_robot = self.get_robot(new_seed)
            candidate.execute(eval_robot)

            log_and_print('switch to robot seed {}'.format(new_seed))

            return self.MORE_WORK_TYPE, eval_robot, np.mean(all_rewards)

    # Add IF (case 1 | expand for one action) or Case 3
    def add_if_branch(self, candidate, eval_reward, eval_robot, store_cost=None):
        robot_seed = eval_robot.seed
        # check break point
        bp_stmts, bp_idx = candidate.find_break_point()
        if bp_stmts is None:
            return

        # get condition
        bp = bp_stmts[bp_idx]
        diff_conds = minigrid_base_dsl.get_diff_conds(bp.abs_state, bp.obs_abs_state)
        random.shuffle(diff_conds)

        # find starting id of IFS before bp
        loc = bp_idx
        while loc - 1 >= 0 and isinstance(bp_stmts[loc - 1], minigrid_base_dsl.IF):
            loc -= 1
        if_start_id = loc

        # add if branch (case 1 | expand for one action)
        case_1_applied = False
        for j in range(len(diff_conds)):
            # insert IF(cond) {C} at break point
            lst = list(range(if_start_id, bp_idx + 1))
            print("lst:", lst)
            random.shuffle(lst)
            print("lst:", lst)
            for k in lst:
                # add IF
                new_if = minigrid_base_dsl.IF(cond=diff_conds[j])
                new_if.stmts[0].touch = True
                bp_stmts.insert(k, new_if)
                tmp_cand = copy.deepcopy(candidate)
                bp_stmts.pop(k)

                # add action
                new_cand_list, new_action_list = tmp_cand.expand_actions(
                    c_touch=True, cond_type="i"
                )
                for new_cand, new_action in zip(new_cand_list, new_action_list):
                    # case one (only once)
                    if str(new_action) == str(bp):
                        if case_1_applied:
                            continue

                        # update abstract state
                        prev_abs_state = bp.abs_state
                        prev_obs_abs_state = bp.obs_abs_state
                        print("merge 0")
                        new_abs_state = minigrid_base_dsl.merge_abs_state(
                            prev_abs_state, bp.obs_abs_state
                        )

                        bp.obs_abs_state = None
                        bp.abs_state = new_abs_state
                        bp.break_point = False
                        new_cand = copy.deepcopy(candidate)
                        new_cand.reset()
                        bp.break_point = True
                        bp.abs_state = prev_abs_state
                        bp.obs_abs_state = prev_obs_abs_state

                        case_1_applied = True

                        # add
                        new_cand.reset_c_touch()
                        new_cand.reset_resume_points()
                        # debug
                        tmp_c_stmts, tmp_c_idx = new_cand.find_actions(c_touch=True)
                        assert tmp_c_stmts is None

                        new_robot = self.get_robot(robot_seed)
                        if store_cost is None:
                            cost, add_success = self.add_queue(
                                new_cand, eval_reward, new_robot
                            )
                        else:
                            cost, add_success = self.add_queue(
                                new_cand,
                                eval_reward,
                                new_robot,
                                cost=store_cost,
                            )

                        # log and print
                        if add_success:
                            log_and_print(
                                "case one applied: put into queue and get cost {} reward {} for\n {}".format(
                                    cost, eval_reward, str(new_cand)
                                )
                            )
                        else:
                            log_and_print(
                                "unable to put into queue and get cost {} reward {} for\n {}".format(
                                    cost, eval_reward, str(new_cand)
                                )
                            )
                        continue

                    # direct add
                    new_cand.reset_c_touch()
                    # debug
                    tmp_c_stmts, tmp_c_idx = new_cand.find_actions(c_touch=True)
                    assert tmp_c_stmts is None

                    new_robot = self.get_robot(robot_seed)
                    if store_cost is None:
                        # pdb.set_trace()
                        if (
                            new_action is not None
                            and new_action.action.action == "return"
                        ):
                            cost, add_success = self.add_queue(
                                new_cand,
                                eval_reward,
                                new_robot,
                                is_return=True,
                            )
                        else:
                            cost, add_success = self.add_queue(
                                new_cand, eval_reward, new_robot
                            )
                    else:
                        # pdb.set_trace()
                        cost, add_success = self.add_queue(
                            new_cand,
                            eval_reward,
                            new_robot,
                            cost=store_cost + 1,
                        )

                    # log and print
                    if add_success:
                        log_and_print(
                            "add if: put into queue and get cost {} reward {} for\n {}".format(
                                cost, eval_reward, str(new_cand)
                            )
                        )
                    else:
                        log_and_print(
                            "unable to put into queue and get cost {} reward {} for\n {}".format(
                                cost, eval_reward, str(new_cand)
                            )
                        )
                    continue

        # case 3: look ahead and put main branch into if branch
        start_end_idx = []
        new_IF_code = []
        new_candidate_code = []

        # TODO: double check from here
        # NOTE: number of ACTION
        len_bp_stmts = len(bp_stmts)  # NOTE: C has been removed in bp_stmts
        len_bp_stmts_effective = 0
        for k in range(bp_idx, len_bp_stmts):
            if isinstance(bp_stmts[k], minigrid_base_dsl.ACTION) or isinstance(
                bp_stmts[k], minigrid_base_dsl.HIDE_ACTION
            ):
                len_bp_stmts_effective += 1

        if len_bp_stmts_effective >= 1:
            start_idx = bp_idx  # starting point of case 3 IF branch, will not change

            # try every end index
            for end_idx in range(start_idx + 1, len_bp_stmts):
                tmp_end_action = bp_stmts[end_idx]
                if not isinstance(tmp_end_action, minigrid_base_dsl.ACTION):
                    continue

                # attempt to execution a single action
                # print('double check state of evaluate robot')
                # pdb.set_trace()
                _eval_robot = self.copy_robot(eval_robot)
                execute_single_action_with_library(tmp_end_action, _eval_robot)
                # if tmp_end_action.action.action in ['get_key', 'get_door', 'get_simple']: #== 'get_key' or tmp_end_action.action.action == 'get_door' or :
                #     action = get_action(tmp_end_action, None, None)
                #     action.execute(_eval_robot, None)
                # else:
                #     _eval_robot.execute_single_action(tmp_end_action.action)
                _eval_robot_current_state = minigrid_base_dsl.get_abs_state(_eval_robot)

                if minigrid_base_dsl.satisfy_abs_state(
                    _eval_robot_current_state, tmp_end_action.post_abs_state
                ):
                    for j in range(len(diff_conds)):
                        # clone
                        _candidate = copy.deepcopy(candidate)
                        _bp_stmts, _bp_idx = _candidate.find_break_point()

                        neg_cond = minigrid_base_dsl.get_neg_cond(diff_conds[j])
                        IF_code = minigrid_base_dsl.IF(cond=neg_cond)
                        IF_code.stmts.pop()  # remove C
                        IF_code.stmts += _bp_stmts[start_idx:end_idx]

                        start_end_idx.append((start_idx, end_idx))
                        new_IF_code.append(IF_code)
                        new_candidate_code.append(_candidate)

            # consider special case when inside if branch (can combine with above case just specify here to mark special case)
            if isinstance(bp_stmts[-1], minigrid_base_dsl.HIDE_ACTION):
                # pdb.set_trace()
                assert len_bp_stmts - bp_idx > 1

                _eval_robot = self.copy_robot(eval_robot)
                execute_single_action_with_library(bp_stmts[-1].action, _eval_robot)
                # _eval_robot.execute_single_action(bp_stmts[-1].action.action)
                _eval_robot_current_state = minigrid_base_dsl.get_abs_state(_eval_robot)

                if minigrid_base_dsl.satisfy_abs_state(
                    _eval_robot_current_state, bp_stmts[-1].action.post_abs_state
                ):
                    for j in range(len(diff_conds)):
                        # clone
                        _candidate = copy.deepcopy(candidate)
                        _bp_stmts, _bp_idx = _candidate.find_break_point()

                        neg_cond = minigrid_base_dsl.get_neg_cond(diff_conds[j])
                        IF_code = minigrid_base_dsl.IF(cond=neg_cond)
                        IF_code.stmts.pop()  # remove C
                        IF_code.stmts += _bp_stmts[start_idx:]

                        start_end_idx.append((start_idx, len_bp_stmts - 1))
                        new_IF_code.append(IF_code)
                        new_candidate_code.append(_candidate)

        assert len(start_end_idx) == len(new_IF_code)
        for k in range(len(new_IF_code)):
            _candidate = new_candidate_code[k]
            _bp_stmts, _bp_idx = _candidate.find_break_point()

            (_start, _end) = start_end_idx[k]
            # _bp_stmts[_bp_idx].break_point = False
            # _bp_stmts[_bp_idx].obs_abs_state = None
            assert new_IF_code[k].stmts[0].break_point is True
            new_IF_code[k].stmts[0].break_point = False
            new_IF_code[k].stmts[0].obs_abs_state = None
            _bp_stmts[_start:_end] = []
            _bp_stmts.insert(_start, new_IF_code[k])
            _candidate.reset()

            # add to queue
            new_robot = self.get_robot(robot_seed)
            if store_cost is None:
                cost, add_success = self.add_queue(
                    _candidate, eval_reward, new_robot
                )
            else:
                cost, add_success = self.add_queue(
                    _candidate,
                    eval_reward,
                    new_robot,
                    cost=store_cost + 1,
                )

            # log print
            if add_success:
                log_and_print(
                    "case 3: put into queue and get cost {} reward {} for\n {}".format(
                        cost, eval_reward, str(_candidate)
                    )
                )
            else:
                log_and_print(
                    "unable to put into queue and get cost {} reward {} for\n {}".format(
                        cost, eval_reward, str(_candidate)
                    )
                )

    def case_two_drop(
        self,
        p,
        tmp_action,
        tmp_abs_state,
        tmp_pos_abs_state,
        bp_idx,
        bp_stmts,
        c_idx,
        c_stmts,
        eval_reward,
        check_IF_id,
        c_cond,
        robot_seed,
        store_cost=None,
    ):
        bp = bp_stmts[bp_idx]
        cand_idx = None
        for idx in range(bp_idx, len(bp_stmts)):
            cand_action = bp_stmts[idx]
            abs_state_check = (
                isinstance(cand_action, minigrid_base_dsl.ACTION)
                and minigrid_base_dsl.satisfy_abs_state(
                    tmp_pos_abs_state, cand_action.post_abs_state
                )
            ) or (
                isinstance(cand_action, minigrid_base_dsl.HIDE_ACTION)
                and minigrid_base_dsl.satisfy_abs_state(
                    tmp_pos_abs_state, cand_action.action.post_abs_state
                )
            )
            action_check = (
                isinstance(cand_action, minigrid_base_dsl.ACTION)
                and str(tmp_action) == str(cand_action)
            ) or (
                isinstance(cand_action, minigrid_base_dsl.HIDE_ACTION)
                and str(tmp_action) == str(cand_action.action)
            )
            if action_check and abs_state_check:
                cand_idx = idx
                break

        # end if or add else
        if cand_idx is not None:
            # debug
            assert bp_stmts[check_IF_id].stmts == c_stmts

            bp.break_point = False
            # remove C
            c_stmts.pop(c_idx)
            # remove duplicate action
            c_stmts.pop(c_idx - 1)
            assert len(c_stmts) == c_idx - 1

            # directly end if
            if cand_idx == bp_idx:
                # update abstract state
                print("merge 1")
                bp.abs_state = minigrid_base_dsl.merge_abs_state(
                    bp.abs_state, tmp_abs_state
                )
                # add hidden action
                hidden_action = minigrid_base_dsl.HIDE_ACTION(bp)
                c_stmts.append(hidden_action)

            # add else branch
            else:
                # update abstract state (double check)
                cand_action = bp_stmts[cand_idx]
                if not isinstance(cand_action, minigrid_base_dsl.HIDE_ACTION):
                    print("merge 2")
                    cand_action.abs_state = minigrid_base_dsl.merge_abs_state(
                        cand_action.abs_state, tmp_abs_state
                    )

                    # add else branch
                    new_if_else_branch = minigrid_base_dsl.IFELSE(c_cond)
                    new_if_else_branch.stmts = bp_stmts[check_IF_id].stmts
                    new_if_else_branch.else_stmts = bp_stmts[check_IF_id + 1 : cand_idx]

                    # debug: try to add additional hidden action
                    new_if_else_branch.stmts.append(
                        minigrid_base_dsl.HIDE_ACTION(cand_action)
                    )
                    new_if_else_branch.else_stmts.append(
                        minigrid_base_dsl.HIDE_ACTION(cand_action)
                    )

                    bp_stmts[check_IF_id:cand_idx] = [new_if_else_branch]

                else:
                    # add else branch
                    print("add else branch")
                    new_if_else_branch = minigrid_base_dsl.IFELSE(c_cond)
                    new_if_else_branch.stmts = bp_stmts[check_IF_id].stmts
                    # debug: try to add additional hidden action
                    new_if_else_branch.stmts.append(cand_action)
                    new_if_else_branch.else_stmts = bp_stmts[check_IF_id + 1 :]

                    bp_stmts[check_IF_id:cand_idx] = [new_if_else_branch]

            # add back
            p.reset_c_touch()
            # debug
            c_stmts, c_idx = p.find_actions(c_touch=True)
            assert c_stmts is None

            p.reset_resume_points()

            new_robot = self.get_robot(robot_seed)
            if store_cost is None:
                cost, add_success = self.add_queue(
                    p, eval_reward, new_robot
                )
            else:
                cost, add_success = self.add_queue(
                    p, eval_reward, new_robot, cost=store_cost - 1
                )

            # log print
            if add_success:
                log_and_print(
                    "case 2: put into queue and get cost {} reward {} for\n {}".format(
                        cost, eval_reward, str(p)
                    )
                )
            else:
                log_and_print(
                    "unable to put into queue and get cost {} reward {} for\n {}".format(
                        cost, eval_reward, str(p)
                    )
                )

        return cand_idx

    # Note:
    # Execute Program:
    # -> Success -> multi. seed -> END
    # -> Fail -> END
    # -> Break Point -> Add IF (case 1 | expand for one action) or Case 3  (should before Find C)
    # -> Find C -> Delete (case 2 | while finish) or Expand (while | IF | drop C (while;C))
    def search(self):
        # debug_store = {}

        cur_id = 0
        while True:
            restart = False
            # do search
            for iter in range(self.max_search_iter):
                log_and_print("[ITER] {}".format(iter))

                # get one program
                if iter == 87: # or iter == 424:
                    # pdb.set_trace()
                    # import programskill.dsl

                    # programskill.dsl.DSL_DEBUG = True
                    a = 1
                    pass
                try:
                    r, ts, p, robot, state, store_cost = self.q.get_nowait()
                except queue.Empty:
                    break
                # double check: debug
                # tmp_c_stmts, _ = p.find_actions(c_touch=True)
                # assert tmp_c_stmts is None
                # if iter in [725]:
                    # self.q = PriorityQueue()
                # log print
                log_and_print("searching base on {} with cost {}".format(str(p), r))

                # Execute Program
                if state is None:
                    # pdb.set_trace()
                    eval_result, eval_robot, reward = self.eval_program(
                        self.seed, robot, p, check_multiple=False
                    )

                    # from matplotlib.pyplot import imsave; imsave("y.png", eval_robot.env.render(mode='rgb_array'))
                    eval_reward = eval_robot.check_reward()
                else:
                    eval_result, eval_reward = state
                    eval_robot = robot
                robot_seed = eval_robot.seed
                p.reset()

                # get action before C
                c_stmts, c_idx = p.find_actions(c_touch=True)
                tmp_action = None
                if c_stmts is not None and len(c_stmts) > 1:
                    tmp_action = c_stmts[c_idx - 1]

                # 1) Success
                if eval_result == self.SUCCESS_TYPE:
                    test_result = self.test_program(p)
                    if self.found_one and test_result == self.SUCCESS_TYPE:
                        # break
                        raise SUCC(p)
                    else:
                        continue

                # 2) Fail
                elif eval_result == self.FAIL_TYPE:
                    continue

                # debug for MORE_WORK
                # if p.check_resume_points():
                #     outer_while_num = len(
                #         [True for code in p.stmts if isinstance(code, WHILE)]
                #     )
                #     if not (
                #         (isinstance(p.stmts[-2], C) and p.stmts[-2].touch)
                #         or (outer_while_num > 1)
                #     ):
                #         pdb.set_trace()
                #     assert (isinstance(p.stmts[-2], C) and p.stmts[-2].touch) or (
                #         outer_while_num > 1
                #     )
                #     p.reset_resume_points()
                # assert not p.check_resume_points()

                # 3) Find Break Point
                bp_stmts, bp_idx = p.find_break_point()
                # check whether if branch has been added
                expand_IF = False
                check_IF_id = None
                if bp_stmts is not None:
                    bp = bp_stmts[bp_idx]
                    for check_id in range(bp_idx - 1, -1, -1):
                        if isinstance(bp_stmts[check_id], minigrid_base_dsl.IF):
                            if self.check_C_exist(bp_stmts[check_id].stmts):
                                expand_IF = True
                                # store
                                assert check_IF_id is None
                                check_IF_id = check_id
                        else:
                            break

                    # case 3 as well as add if
                    if not expand_IF:
                        # TODO: double check
                        # assert c_stmts is None
                        self.add_if_branch(
                            copy.deepcopy(p),
                            eval_reward,
                            self.copy_robot(eval_robot),
                            store_cost=store_cost,
                        )
                        continue

                # 4) Find C
                c_cond, cond_type = p.find_c_cond(c_touch=True)
                if c_stmts is None:
                    if expand_IF:
                        log_and_print(
                            "invalid if branch added and drop for now \n {}".format(
                                str(p)
                            )
                        )
                        continue
                    else:
                        # pdb.set_trace()
                        log_and_print(
                            "Special case appear and drop for now\n {}".format(str(p))
                        )
                        continue

                # drop if main branch cannot be entered
                if iter == 0 and cond_type is None:
                    # pdb.set_trace()
                    log_and_print("main branch cannot be entered and drop\n")
                    # break

                # for debug
                if expand_IF:
                    if cond_type != "i":
                        continue
                        pdb.set_trace()
                    assert cond_type == "i"

                # Delete (case 2)
                # consider special
                # TODO: can use smarter way to handle, now just skip it
                if expand_IF and tmp_action.abs_state is None:
                    log_and_print("should not get here now")
                    pdb.set_trace()
                    log_and_print(
                        "location (b): Special case appear and drop for now\n {}".format(
                            str(p)
                        )
                    )
                    continue

                # if expand_IF and len(c_stmts) > 2:
                #     # find all satisfied actions
                #     pdb.set_trace()
                #     print('should not get here')
                #     cand_idx = self.case_two_drop(p, tmp_action, tmp_action.abs_state, tmp_action.post_abs_state, bp_idx, bp_stmts, c_idx, c_stmts, eval_reward)
                #     if cand_idx is not None:
                #         continue

                # Delete (while finish)
                action_num = 0
                for code in c_stmts:
                    if isinstance(code, minigrid_base_dsl.ACTION):
                        action_num += 1

                if (
                    cond_type == "w"
                    and eval_robot.execute_single_cond(c_cond)
                    and action_num > 0
                ):
                    w_stmt = p.find_c_stmt(cond_type="w", c_touch=True)
                    # TODO: check whether use
                    if w_stmt.robot_move:
                        # resume point
                        # pdb.set_trace()
                        print("should not get here for while finish")
                        # return
                        continue

                # Expand (drop when while;C)
                # set resume point
                # p.set_resume_points()

                # expand
                p_list, action_list = p.expand_actions(
                    c_touch=True, cond_type=cond_type
                )
                index = list(range(len(p_list)))
                if self.shuffle_actions:
                    np.random.shuffle(index)

                # for either IF or WHILE
                for i in index:
                    # get action and program
                    cur_action = action_list[i]
                    candidate = p_list[i]

                    # check if case two apply
                    if cur_action is not None and cur_action.action.action != "return":
                        if expand_IF:
                            c_stmts, c_idx = candidate.find_actions(c_touch=False)
                            bp_stmts, bp_idx = candidate.find_break_point()
                        if expand_IF and len(c_stmts) > 2:
                            # execute once
                            new_robot = self.copy_robot(eval_robot)
                            execute_single_action_with_library(cur_action, new_robot)
                            # new_robot.execute_single_action(cur_action.action)
                            cur_abs_state = minigrid_base_dsl.get_abs_state(new_robot)
                            prev_abs_state = minigrid_base_dsl.get_abs_state(eval_robot)
                            new_robot.check_reward()
                            # drop
                            cand_idx = self.case_two_drop(
                                candidate,
                                cur_action,
                                prev_abs_state,
                                cur_abs_state,
                                bp_idx,
                                bp_stmts,
                                c_idx,
                                c_stmts,
                                eval_reward,
                                check_IF_id,
                                c_cond,
                                robot_seed
                            )
                            if cand_idx is not None:
                                continue
                        if not expand_IF and cond_type == "w":
                            # execute once
                            new_robot = self.copy_robot(eval_robot)
                            execute_single_action_with_library(cur_action, new_robot)
                            # new_robot.execute_single_action(cur_action.action)
                            w_stmt = candidate.find_c_stmt(cond_type="w", c_touch=False)
                            if new_robot.execute_single_cond(c_cond):
                                # pdb.set_trace()
                                print(type(new_robot.initial_pos), type(new_robot.env.agent_pos))
                                if (
                                    new_robot.initial_pos != new_robot.env.agent_pos
                                ).any():
                                    # only finish the while when the agent has moved

                                    c_stmts, c_idx = candidate.find_actions(
                                        c_touch=False
                                    )
                                    c_stmts.pop(c_idx)

                    candidate.reset_c_touch()
                    # debug
                    c_stmts, c_idx = candidate.find_actions(c_touch=True)
                    assert c_stmts is None
                    # add back
                    # new_robot = self.copy_robot(eval_robot)
                    new_robot = self.get_robot(robot_seed)
                    new_robot.active = True

                    if store_cost is None:
                        if (
                            cur_action is not None
                            and cur_action.action.action == "return"
                        ):
                            cost, add_success = self.add_queue(
                                candidate, eval_reward, new_robot, is_return=True
                            )
                        else:
                            cost, add_success = self.add_queue(
                                candidate, eval_reward, new_robot
                            )
                    else:
                        if cur_action is None:
                            new_cost = store_cost
                        else:
                            new_cost = store_cost + 1
                        cost, add_success = self.add_queue(
                            candidate, eval_reward, new_robot, cost=new_cost
                        )
                    # if cur_action is None:
                    #     cost, add_success = self.add_queue(candidate, eval_reward, new_robot, cost=store_cost)
                    # else:
                    #     cost, add_success = self.add_queue(candidate, eval_reward, new_robot, cost=store_cost+1)
                    # cost, add_success = self.add_queue(candidate, eval_reward, self.get_robot(self.seed))

                    # log print
                    if cur_action is None:
                        log_and_print("attempt to drop:")
                    if add_success:
                        log_and_print(
                            "expand: put into queue and get cost {} reward {} for\n {}".format(
                                cost, eval_reward, str(candidate)
                            )
                        )
                    else:
                        log_and_print(
                            "unable to put into queue and get cost {} reward {} for\n {}".format(
                                cost, eval_reward, str(p)
                            )
                        )
                    continue

                # log print
                log_and_print("------ end current search -------\n")

            if not restart:
                print("[success programs]")
                # pdb.set_trace()
                count = 0
                for reward, program in self.candidates["success"]:
                    print(program.to_string_verbose())
                    count += 1
                if count > 0:
                    print("succeed")
                print("====== stop counting environment interactions ==========")
                print(
                    f"number of environment interactions is {minigrid_implement.dsl.search_counter}"
                )
                minigrid_implement.dsl.SEARCH_STATUS = False
                # for reward, program in self.candidates["success"]:
                #     print("====== removing abs state ========")
                #     program.remove_abs_state()
                #     print(program.to_string_verbose())
                #     print("====== execute and update ========")
                #     for e in self.eval_seeds:
                #         robot = self.get_robot(e)
                #         program.execute_and_update(robot)
                #         print(f"seed {e} completed", flush=True)

                #     print("====== final program =======")
                #     print(program.to_string_verbose())
                #     import pickle

                #     with open("search_saved_program", "wb") as f:
                #         pickle.dump(program, f)
                break


# TODO: 1) set structural cost limit [done]
#       2-1) case 1 [done]
#       2-2) case 2 [done]
#       2-3) case 3 [done]
#       2) N = 3 test cases for success program [done]
#       3) multiple C in a sketch [done]
#       4) check chain of rules [done]
#       5) candidate.execute(eval_robot), stop when reward is 1 or -1 [done]
#       6) shuffle WRAPPED_ACTION_LIST / COND_LIST [done]
#       7) topOff [done*]
#       8) stairClimber [done*]
#       9) fourCorner [done*]
#      10) randomMaze ----> generator is not fixed?

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="store", default="search")
    parser.add_argument("--make_video", action="store_const", const=True, default=False)
    parser.add_argument("--env", action="store", required=True)
    parser.add_argument("--seed", action="store", required=True)
    parser.add_argument("--plot_env", action="store_const", const=True, default=False)
    parser.add_argument("--random_seed", action="store", required=True)
    args = parser.parse_args()
    random.seed(int(args.random_seed))
    np.random.seed(int(args.random_seed))

    minigrid_base_dsl.set_action_dict(args.env)
    minigrid_base_dsl.set_cond_dict(args.env)
    # minigrid_base_dsl.ABS_STATE = minigrid_base_dsl.set_abs_state(args.env)

    # NOTE: for simplicity, not a tree right now
    program_db = []
    # pdb.set_trace()

    p = minigrid_base_dsl.Program()
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
    # pdb.set_trace()


    # seed = int(sys.argv[1])
    seed = int(args.seed)
    # more_seeds = []
    # eval_seeds = [seed]
    # more_seeds = [28, 36]
    # more_seeds = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
    # more_seeds = [i for i in range(2, 5)]
    # eval_seeds = [i for i in range(0, 10)]
    # more_seeds = [1]
    eval_seeds = [i for i in range(0, 5000)]    
    more_seeds = [i for i in range(1, 50)]
    # eval_seeds = [i for i in range(1, 10)]
    # eval_seeds = [4845]

    if args.plot_env:
        for i in range(0, 1500):
            robot = Robot(args.env, i)
            robot.env.env.env.render(f"initial/{i}.png")
        exit()
    # robot2 = copy_robot(robot)
    # program = program_db[6].expand()[0].expand()[0]
    if args.env is None:
        print("environment not provided")
    elif args.env == "MiniGrid-RandomCrossingS11N5-v0":
        program = (
            program_db[12].expand()[0].expand()[2].expand()[11].expand()[0].expand()[0]
        )
    elif args.env == "MiniGrid-MultiRoomNoDoor-N6-v0":
        program = program_db[17].expand()[0].expand()[0]
    elif args.env == "MiniGrid-MultiRoom-N6-v0":
        with open("MiniGrid-MultiRoomNoDoor-N6-v0_saved_program", "rb") as f:
            program = pickle.load(f)
    elif args.env == "MiniGrid-LockedRoom-v0":
        with open("MiniGrid-MultiRoom-N6-v0_saved_program_for_lockedroom", "rb") as f:
            # with open("current/tree/MiniGrid-MultiRoom-N6-v0_saved_program_for_lockedroom", "rb") as f:
            program = pickle.load(f)
    elif args.env == "MiniGrid-DoorKey-6x6-v0":
        with open("MiniGrid-LockedRoom-v0_saved_program", "rb") as f:
            program = pickle.load(f)
    elif args.env == "MiniGrid-DoorKey-8x8-v0":
        # with open("current/tree/MiniGrid-LockedRoom-v0_saved_program", "rb") as f:
        with open("MiniGrid-LockedRoom-v0_saved_program", "rb") as f:
            program = pickle.load(f)
    elif args.env == "MiniGrid-UnlockPickup-v0":
        program = program_db[11].expand()[0].expand()[0]
        # program = program_db[2].expand()[8].expand()[0].expand()[0]
        # pdb.set_trace()
    elif args.env == "MiniGrid-KeyCorridorS3R1-v0":
        pass
    elif args.env == "MiniGrid-RandomLavaCrossingS11N5-v0":
        # with open("MiniGrid-RandomCrossingS11N5-v0_saved_program_for_lava", "rb") as f:
        # program = pickle.load(f)
        program = (
            program_db[14].expand()[0].expand()[2].expand()[12].expand()[0].expand()[0]
        )
        # program = debug_program.convert_program("WHILE(not (goal_on_right)) { IF(left_is_clear) { turn_left}  IF(goal_present) { return}  IF(not (front_is_clear)) { turn_right}  move} turn_right WHILE(not (goal_present)) { move}")
    else:
        print("environment not found")
    # debug program
    # debug_prog_str = "WHILE(not(front_is_wall)) { IF(front_is_obj) { turn_right}  move} turn_right WHILE(not (goal_on_right)) {   IF(left_is_clear) { turn_left}   IF(goal_present) { return} IF(front_is_closed_door) { IF(front_is_locked_door) { get_key pickup get_locked_door} toggle} IF(not (front_is_clear)) { turn_right} move} turn_right WHILE(not (goal_present)) { move} "
    # debug_prog_str = "WHILE(not(front_is_wall)) { IF(front_is_obj) { turn_right}  move} turn_right WHILE(not (goal_on_right)) {   IF(left_is_clear) { turn_left} IF(goal_present) { return}  IF(front_is_closed_door) { toggle} IF(not (front_is_clear)) { turn_right} move} turn_right WHILE(not (goal_present)) { move} END"
    # debug_prog_str = "get_key pickup get_locked_door toggle get_goal"
    debug_prog_str = "WHILE(front_is_clear) { IF(front_is_closed_door) { turn_right}  move} get_box IF(right_is_clear) {turn_right drop turn_left} ELSE {turn_left drop turn_right} pickup"
    debug_prog_str = "WHILE(not (goal_on_right)) { IF(left_is_clear) { turn_left}  IF(goal_present) { return}  IF(not (front_is_clear)) { turn_right}  move} turn_right WHILE(not (goal_present)) { move}"
    # debug_prog_str = "WHILE(not (goal_on_right)) { IF(left_is_clear) { IF(not (goal_present)) { turn_left} }  IF(not (front_is_clear)) { turn_right}  IF(goal_present) { turn_left}  move} turn_right WHILE(not (goal_present)) { move}"
    debug_prog_str = "WHILE(not (goal_on_right)) { IF(left_is_clear) { turn_left}  IF(goal_present) { return}  IF(not (front_is_clear)) { turn_right}  IF(front_is_lava) { turn_right}  move} turn_right WHILE(not (goal_present)) { move}"
    debug_prog_str = "WHILE(not (goal_on_right)) { IF(front_is_clear) { IF(not (left_is_clear)) { IF(goal_present) { turn_right }  move}  turn_left} ELSE { IF(not (goal_present)) { turn_right} } move}  turn_right WHILE(not (goal_present)) { move}"
    debug_prog_str = "get_ball"
    if args.task == "test":
        program = debug_program.convert_program(debug_prog_str)
        # with open("doorkey_program", "rb")  as f:
        #     program = pickle.load(f)
        # program = lockedroom_get("goal")
        # pdb.set_trace()
    print("startin from program ", program)
    print(program.to_string_verbose())
    # exit()

    node = Node(
        sketch=program,
        task=args.env,
        seed=seed,
        more_seeds=more_seeds,
        eval_seeds=eval_seeds,
        max_search_iter=700000,
        max_structural_cost=20,
        shuffle_actions=True,
        found_one=True,
        make_video=args.make_video,
    )

    if args.make_video:
        # remove old files
        files = os.listdir("frames/")
        for f in files:
            os.remove(os.path.join("frames", f))

        import minigrid_implement.dsl

        minigrid_implement.dsl.DSL_DEBUG = True

    if args.task == "search":
        node.search()
    elif args.task == "test":
        print("enter")
        # program = get("goal")
        # print(program)
        # program = node.remove_and_update(program)
        # save_prog(program, "multiroom_program")
        # pdb.set_trace()
        rate = node.test_for_success_rate(program)
        from termcolor import colored

        print(colored(f"success rate is {rate * 100}%", "light_yellow"))
    elif args.task == "execute":
        program = minigrid_base_dsl.lockedroom_get("goal")
        print("starting to update abs of ")
        print(program)
        program = node.remove_and_update(program)
        save_prog(program, f"{args.env}_saved_program_for_lava")

    if args.make_video:
        os.system("bash make_video.sh")
