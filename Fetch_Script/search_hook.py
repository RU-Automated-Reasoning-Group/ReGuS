import argparse
import copy
# NOTE: each node is associated with a sketch
import enum
import multiprocessing as mp
import os
import pdb
import queue
import random
import sys
import time
from queue import PriorityQueue

import numpy as np
import pandas as pd
# import programskill.programskill.reskill
import reskill

from programskill.robot_hook import SkillRobot

# using pick and place environment
import programskill.hook_dsl as dsl
from robot_hook_dsl import *

from utils.logging import log_and_print



# sys.path.append('yourpath/StructuredProgramSearch/current/tree/programskill/programskill/')
df = None

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
    for s in ACTION_DICT:
        cost += program_str.count(s)

    cost += program_str.count("return")
    # cost += program_str.count("IF") * 0.1
    cost += program_str.count("get_simple")
    return cost


def execute_single_action_with_library(action, robot):
    # action is a action object
    if action.action.action in ["pick_up_hook"]:
        action = get_action(action, None, None)
        action.execute(robot, None)
        robot.active = True
    else:
        robot.execute_single_action(action.action)

def execute_for_reward(cur_robot, program, force=True):
    cur_prog = copy.deepcopy(program)
    cur_prog.reset()
    cur_robot.force_execution = force

    cur_prog.execute(cur_robot)

    return cur_robot.no_fuel(), cur_robot.check_reward(), cur_prog.complete()

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
        logic_expr=None,
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
        self.logic_expr = logic_expr

        # store all required robot
        self.robot_store = {self.seed: SkillRobot(self.task, seed=self.seed)}
        # add environment specific actions here
        # robot = MiniGridRobot(self.task, seed=self.seed)
        # mission_str = robot.env.env.mi
        for e in more_seeds:
            self.robot_store[e] = SkillRobot(self.task, seed=e)
        for e in eval_seeds:
            self.robot_store[e] = SkillRobot(self.task, seed=e)
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

    # used for single simulator
    def get_robot(self, seed):
        return SkillRobot(self.task, seed)

    def copy_robot(self, robot):
        robot_copy = copy.deepcopy(robot)
        # robot_copy.env.sim.set_state(copy.deepcopy(robot.env.sim.get_state()))
        # robot_copy.env.sim.forward()
        robot_copy.env.env.goal = copy.deepcopy(robot.env.goal)
        robot_copy.env.env.initial_state = copy.deepcopy(robot.env.initial_state)
        robot_copy.env.env.obs = copy.deepcopy(robot.env.obs)
        robot_copy.env.env.sim_state = copy.deepcopy(robot.env.sim_state)
        # robot_copy.env.reward_info = 'place'
        return robot_copy

    # add candidate program into queue
    def add_queue(self, candidate, reward, robot, cost=None, is_return=False):
        # calculate reward before add
        print("trying to add", candidate)
        eval_result, robot = self.eval_program(
            self.seed, robot, candidate, check_multiple=False
        )
        reward = robot.check_reward()
        if is_return and reward != 1:
            return cost, False
        # add back

        # cover the case that the last while loop is only used once
        if (
            candidate.count_C() <= 2
            and type(candidate.stmts[-2]) == C
            and robot.reward == 1
        ):
            reward = 1
            eval_result = self.SUCCESS_TYPE

        # assert cost is None
        if cost is None:
            cost = get_structural_cost(candidate)
        
        if reward == 1:
            # now just evaluate in one thread due to the implementation of the fetch environment
            # result = []
            # for e in self.more_seeds:
            #     r = execute_for_reward(self.get_robot(e), candidate, Fa)
            # results = [self.pool.apply_async(execute_for_reward, args=(self.get_robot(e), candidate, True)) \
            #             for e in self.more_seeds]
            
            # single thread version, used for plot
            results = []
            for e in self.more_seeds:
                results.append(execute_for_reward(self.get_robot(e), candidate, True))
            # results = [p.get() for p in results]
            all_rewards = [x[1] for x in results]
            reward = np.mean(all_rewards)
            for idx, s in enumerate(self.more_seeds):
                print(f"seed {s} with reward {all_rewards[idx]}")
            print(f"reward updated from 1 to {reward}")

        q_tuple = (
            # -reward + cost * 0.02,
            -reward + cost * 0.5,
            # -reward + cost * 0.1,
            # -reward + cost * 0.2,
            time.time(),
            candidate,
            robot,
            (eval_result, reward),
            None,
        )
        # pdb.set_trace()
        # for plot, not used in training
        dsl.stop_count()
        visualization_rwd = []
        if reward == 0.5:
            print("sampling 3 seeds")
            sampled_seeds = random.sample(self.eval_seeds, 3)
        elif reward == 1.0:
            print("sampling 300 seeds")
            sampled_seeds = random.sample(self.eval_seeds, 300)
        else:
            sampled_seeds = []
            visualization_rwd.append(0.0)
        for e in sampled_seeds:
            # if e % 100 == 0:
            #     print(f"evaluating seed {e}")
            visualization_rwd.append(execute_for_reward(self.get_robot(e), candidate, True)[1])
        mean_rwd = np.mean(visualization_rwd)
        print(f"mean reward is {mean_rwd}")
        df2 = pd.DataFrame([[dsl.get_g_counter(), mean_rwd]], columns=["Step", "Value"])
        # # pdb.set_trace()
        global df
        df = df.append(df2)
        dsl.start_count()
        # print(df)
        if not cost > self.max_structural_cost:
            self.q.put(q_tuple)
            print(f"equeued reward {reward} cost {cost} final {q_tuple[0]}")
            return cost, True
        return cost, False

    # check whether C exist in statement
    def check_C_exist(self, stmts):
        exist = False
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                pass
            elif isinstance(code, (WHILE, IF, IFELSE)):
                exist = self.check_C_exist(code.stmts)
                if exist:
                    return exist
            elif isinstance(code, C):
                return True
            else:
                raise ValueError("Invalide code")

        return exist

    def test_program(self, prog):
        dsl.stop_count()
        success_count = 0
        if self.logic_expr:
            for e in self.eval_seeds:
                # force evaluate
                force_eval_robot = self.get_robot(e)
                force_eval_robot.force_execution = True
                prog.execute(force_eval_robot)
                prog.reset()
                force_eval_robot.force_execution = False
                passed = self.logic_expr(force_eval_robot) and (
                    not force_eval_robot.no_fuel()
                )
                if passed:
                    print(f"seed {e} passed")
                else:
                    print(f"seed {e} failed")
                # fail
                if not passed:
                    # log and print
                    log_and_print(
                        "\nfound but not success in all seeds for \n {}".format(prog)
                    )
                    print("location 0")
                    print(f"seed {str(e)} failed")
                    self.candidates["success_search"].append((1, prog))
                    return self.FAIL_TYPE
            # success
            log_and_print("\nsuccess and store for \n {}".format(prog))
            self.candidates["success"].append((1, prog))
            return self.SUCCESS_TYPE
        
        for e in self.eval_seeds:
            # force evaluate
            force_eval_robot = self.get_robot(e)
            force_eval_robot.force_execution = True
            prog.execute(force_eval_robot)
            prog.reset()
            force_eval_robot.force_execution = False
            passed = force_eval_robot.env.env.get_reward() == 1
            if passed:
                success_count += 1
                print(f"seed {e} passed", flush=True)
            else:
                print(f"seed {e} failed", flush=True)
            # fail
            # if not passed:
            #     # log and print
            #     log_and_print(
            #         "\nfound but not success in all seeds for \n {}".format(prog)
            #     )
            #     # print("successful rate", self.test_for_success_rate(prog))
            #     print("location 1")
            #     print(f"seed {str(e)} failed")
            #     self.candidates["success_search"].append((1, prog))
            #     return self.FAIL_TYPE
        # fail
        dsl.start_count()
        success_rate = success_count / len(self.eval_seeds)
        print(f"success rate is {success_rate}")
        if success_rate < 0.9:
            log_and_print(
                "\nfound but not success in all seeds for \n {}".format(prog)
            )
            return self.FAIL_TYPE

        # success
        log_and_print("\nsuccess and store for \n {}".format(prog))
        self.candidates["success"].append((1, prog))
        df2 = pd.DataFrame([[dsl.get_g_counter(), success_rate]], columns=["Step", "Value"])
        global df
        df = df.append(df2)
        
        return self.SUCCESS_TYPE

    def test_for_success_rate(self, prog):
        count = 0
        # lst = [self.seed]
        lst = self.eval_seeds
        for e in lst:
            print(f"============== seed {e} ==============")
            # force evaluate
            force_eval_robot = SkillRobot(self.task, e)
            force_eval_robot.force_execution = True
            prog.execute(force_eval_robot)
            prog.reset()
            force_eval_robot.force_execution = False
            # if force_eval_robot.check_success():
            if force_eval_robot.env.env.get_reward() == 1:
                count += 1
                print(f"seed {e} success")
            else:
                print(f"seed {e} fail")
        # print(f"count: {count}, len {len(self.eval_seeds)}")
        return count / len(lst)

    def test_success_rate_env(self, num_of_seeds, env_name, prog):
        count = 0
        for i in range(0, num_of_seeds):
            force_eval_robot = SkillRobot(env_name, i, 2000)
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

    # def eval_program(self, seed, robot, candidate, check_multiple=False):
    #     single_seed = len(self.more_seeds) == 0
    #     check_multiple = False if single_seed else check_multiple

    #     # execute and get reward
    #     # eval_robot = self.get_robot(seed)
    #     eval_robot = robot
    #     candidate.execute(eval_robot)
    #     r = eval_robot.check_reward()
    #     complete = candidate.complete()

    #     # update r
    #     # if r >= 0 and not eval_robot.no_fuel():
    #     #     r = 0

    #     # fail cases
    #     # fail
    #     if r == -1:
    #         # log and print
    #         log_and_print('\n fail for \n {}'.format(candidate))
    #         self.candidates['failed'].append((-1, candidate))
    #         return self.FAIL_TYPE, eval_robot, -1
    #     # complete
    #     elif complete:
    #         # log and print
    #         log_and_print('\n complete for\n {}'.format(candidate))
    #         self.candidates['complete'].append((r, candidate))
    #         return self.FAIL_TYPE, eval_robot, -1

    #     # success or more work
    #     all_rewards = [r]
    #     all_no_fuel = [eval_robot.no_fuel()]
    #     # success_seeds = [seed]
    #     all_seeds = [self.seed] + self.more_seeds
    #     all_seeds.pop(all_seeds.index(eval_robot.seed))
    #     log_and_print ('self.seed = {}, seed = {}, eval_robot.seed = {}'.format(self.seed, seed, eval_robot.seed))

    #     # prepare processes
    #     results = [self.pool.apply_async(execute_for_reward, args=(self.get_robot(e), candidate, False)) \
    #                    for e in all_seeds]
    #     results = [p.get() for p in results]

    #     for robot_no_fuel, reward, complete in results:
    #         # C or breakpoint
    #         # if reward >= 0 and not robot_no_fuel:
    #         #     all_rewards.append(0)
    #         # else:
    #         all_rewards.append(reward)
    #         all_no_fuel.append(robot_no_fuel)

    #     log_and_print('all seeds {}'.format(all_seeds))
    #     log_and_print('all rewards: {}'.format(all_rewards))
    #     log_and_print('all no_fuel: {}'.format(all_no_fuel))

    #     # check fail
    #     for robot_no_fuel, reward, complete in results:
    #         if reward == -1:
    #             # log and print
    #             log_and_print('\n fail for \n {}'.format(candidate))
    #             self.candidates['failed'].append((-1, candidate))
    #             return self.FAIL_TYPE, eval_robot, -1
    #         elif complete:
    #             # log and print
    #             pdb.set_trace()
    #             log_and_print('\n complete for\n {}'.format(candidate))
    #             self.candidates['complete'].append((r, candidate))
    #             return self.FAIL_TYPE, eval_robot, -1

    #     # more work on search seed
    #     if not eval_robot.no_fuel():
    #         if candidate.count_C() == 1 and isinstance(candidate.stmts[-2], C):
    #             # pdb.set_trace()
    #             log_and_print('\n special complete with reward {} for\n {}'.format(np.mean(all_rewards), candidate))
    #             self.candidates['complete'].append((np.mean(all_rewards), candidate))

    #         return self.MORE_WORK_TYPE, eval_robot, np.mean(all_rewards)

    #     # check success
    #     if len(all_no_fuel) == np.sum(all_no_fuel):
    #         # log and print
    #         log_and_print('\n no fuel with reward {} for \n {}'.format(np.mean(all_rewards), candidate))
    #         self.candidates['no_fuel'].append((np.mean(all_rewards), candidate))

    #         return self.FAIL_TYPE, eval_robot, np.mean(all_rewards)
    #     else:
    #         log_and_print('\nfound but not success in all seeds with reward {} for \n {}'.format(np.mean(all_rewards), candidate))
    #         self.candidates['success_search'].append((1, candidate))

    #         new_seed = all_seeds[all_no_fuel.index(False)-1]
    #         candidate.reset()
    #         eval_robot = self.get_robot(new_seed)
    #         candidate.execute(eval_robot)

    #         log_and_print('switch to robot seed {}'.format(new_seed))

    #         return self.MORE_WORK_TYPE, eval_robot, np.mean(all_rewards)

    def eval_program(self, seed, robot, candidate, check_multiple):
        single_seed = len(self.more_seeds) == 0
        check_multiple = False if single_seed else check_multiple

        # execute and get reward
        # eval_robot = self.get_robot(seed)
        eval_robot = robot
        # eval_robot.force_execution = True
        candidate.execute(eval_robot)
        r = eval_robot.check_reward()
        true_r = eval_robot.env.env.get_reward()

        # success
        if true_r == 1:
            # multiple seed check
            if check_multiple:
                passed = True
                for e in self.more_seeds:
                    # force evaluate
                    force_eval_robot = self.get_robot(e)
                    force_eval_robot.force_execution = True
                    candidate.reset()
                    candidate.reset_resume_points()
                    candidate.execute(force_eval_robot)
                    candidate.reset()
                    force_eval_robot.force_execution = False
                    passed = passed and force_eval_robot.env.env.get_reward() == 1
                    # attempt to add
                    if not passed:
                        # log and print
                        log_and_print(
                            "\nfound but not success in all seeds for \n {}".format(
                                candidate
                            )
                        )
                        # print("successful rate", self.test_for_success_rate(candidate))
                        print(f"seed {str(e)} failed")
                        print("location 3")
                        self.candidates["success_search"].append((1, candidate))
                        # TODO: double check
                        # pdb.set_trace()
                        candidate.reset()
                        candidate.reset_resume_points()
                        eval_result, eval_robot = self.eval_program(
                            e, self.get_robot(e), candidate, check_multiple=False
                        )
                        assert eval_result != self.SUCCESS_TYPE
                        # pdb.set_trace()
                        return eval_result, eval_robot
            else:
                # avoid insert duplicate programs
                success = True
                if single_seed:
                    # pdb.set_trace()
                    log_and_print(
                        "\n success when not check multi for \n {}".format(candidate)
                    )
                    self.candidates["success"].append((1, candidate))

            return self.SUCCESS_TYPE, eval_robot
        # fail
        elif r == -1:
            # log and print
            log_and_print("\n fail for \n {}".format(candidate))
            self.candidates["failed"].append((-1, candidate))
            return self.FAIL_TYPE, eval_robot
        # no fuel
        elif eval_robot.no_fuel():
            # log and print
            log_and_print("\n no fuel with reward {} for \n {}".format(r, candidate))
            self.candidates["no_fuel"].append((r, candidate))
            return self.FAIL_TYPE, eval_robot
        # complete
        elif candidate.complete():
            # log and print
            log_and_print("\n complete with reward {} for\n {}".format(r, candidate))
            self.candidates["complete"].append((r, candidate))
            return self.FAIL_TYPE, eval_robot
        # need additional operation
        else:
            if candidate.count_C() == 1 and isinstance(candidate.stmts[-2], C):
                log_and_print(
                    "\n special complete with reward {} for\n {}".format(r, candidate)
                )
                self.candidates["complete"].append((r, candidate))

            return self.MORE_WORK_TYPE, eval_robot

    # Add IF (case 1 | expand for one action) or Case 3
    def add_if_branch(self, candidate, eval_reward, eval_robot, store_cost=None):
        # check break point
        bp_stmts, bp_idx = candidate.find_break_point()
        if bp_stmts is None:
            return

        # get condition
        bp = bp_stmts[bp_idx]
        diff_conds = get_diff_conds(bp.abs_state, bp.obs_abs_state)

        # find starting id of IFS before bp
        loc = bp_idx
        while loc - 1 >= 0 and isinstance(bp_stmts[loc - 1], IF):
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
                new_if = IF(cond=diff_conds[j])
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
                        new_abs_state = merge_abs_state(
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
                        # debug
                        tmp_c_stmts, tmp_c_idx = new_cand.find_actions(c_touch=True)
                        assert tmp_c_stmts is None

                        if store_cost is None:
                            cost, add_success = self.add_queue(
                                new_cand, eval_reward, self.get_robot(self.seed)
                            )
                        else:
                            cost, add_success = self.add_queue(
                                new_cand,
                                eval_reward,
                                self.get_robot(self.seed),
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

                    if store_cost is None:
                        # pdb.set_trace()
                        if (
                            new_action is not None
                            and new_action.action.action == "return"
                        ):
                            cost, add_success = self.add_queue(
                                new_cand,
                                eval_reward,
                                self.get_robot(self.seed),
                                is_return=True,
                            )
                        else:
                            cost, add_success = self.add_queue(
                                new_cand, eval_reward, self.get_robot(self.seed)
                            )
                    else:
                        # pdb.set_trace()
                        cost, add_success = self.add_queue(
                            new_cand,
                            eval_reward,
                            self.get_robot(self.seed),
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
            if isinstance(bp_stmts[k], ACTION) or isinstance(bp_stmts[k], HIDE_ACTION):
                len_bp_stmts_effective += 1

        if len_bp_stmts_effective >= 1:
            start_idx = bp_idx  # starting point of case 3 IF branch, will not change

            # try every end index
            for end_idx in range(start_idx + 1, len_bp_stmts):
                tmp_end_action = bp_stmts[end_idx]
                if not isinstance(tmp_end_action, ACTION):
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
                _eval_robot_current_state = get_abs_state(_eval_robot)

                if satisfy_abs_state(
                    _eval_robot_current_state, tmp_end_action.post_abs_state
                ):
                    for j in range(len(diff_conds)):
                        # clone
                        _candidate = copy.deepcopy(candidate)
                        _bp_stmts, _bp_idx = _candidate.find_break_point()

                        neg_cond = get_neg_cond(diff_conds[j])
                        IF_code = IF(cond=neg_cond)
                        IF_code.stmts.pop()  # remove C
                        IF_code.stmts += _bp_stmts[start_idx:end_idx]

                        start_end_idx.append((start_idx, end_idx))
                        new_IF_code.append(IF_code)
                        new_candidate_code.append(_candidate)

            # consider special case when inside if branch (can combine with above case just specify here to mark special case)
            if isinstance(bp_stmts[-1], HIDE_ACTION):
                # pdb.set_trace()
                assert len_bp_stmts - bp_idx > 1

                _eval_robot = self.copy_robot(eval_robot)
                execute_single_action_with_library(bp_stmts[-1].action, _eval_robot)
                # _eval_robot.execute_single_action(bp_stmts[-1].action.action)
                _eval_robot_current_state = get_abs_state(_eval_robot)

                if satisfy_abs_state(
                    _eval_robot_current_state, bp_stmts[-1].action.post_abs_state
                ):
                    for j in range(len(diff_conds)):
                        # clone
                        _candidate = copy.deepcopy(candidate)
                        _bp_stmts, _bp_idx = _candidate.find_break_point()

                        neg_cond = get_neg_cond(diff_conds[j])
                        IF_code = IF(cond=neg_cond)
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
            assert new_IF_code[k].stmts[0].break_point == True
            new_IF_code[k].stmts[0].break_point = False
            new_IF_code[k].stmts[0].obs_abs_state = None
            _bp_stmts[_start:_end] = []
            _bp_stmts.insert(_start, new_IF_code[k])
            _candidate.reset()

            # add to queue
            if store_cost is None:
                cost, add_success = self.add_queue(
                    _candidate, eval_reward, self.get_robot(self.seed)
                )
            else:
                cost, add_success = self.add_queue(
                    _candidate,
                    eval_reward,
                    self.get_robot(self.seed),
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
        store_cost=None,
    ):
        bp = bp_stmts[bp_idx]
        cand_idx = None
        for idx in range(bp_idx, len(bp_stmts)):
            cand_action = bp_stmts[idx]
            abs_state_check = (
                isinstance(cand_action, ACTION)
                and satisfy_abs_state(tmp_pos_abs_state, cand_action.post_abs_state)
            ) or (
                isinstance(cand_action, HIDE_ACTION)
                and satisfy_abs_state(
                    tmp_pos_abs_state, cand_action.action.post_abs_state
                )
            )
            action_check = (
                isinstance(cand_action, ACTION) and str(tmp_action) == str(cand_action)
            ) or (
                isinstance(cand_action, HIDE_ACTION)
                and str(tmp_action) == str(cand_action.action)
            )
            if action_check and abs_state_check:
                # if str(tmp_action) == str(bp_stmts[bp_idx]) and abs_state_check:
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
                bp.abs_state = merge_abs_state(bp.abs_state, tmp_abs_state)
                # add hidden action
                hidden_action = HIDE_ACTION(bp)
                c_stmts.append(hidden_action)

            # add else branch
            else:
                # update abstract state (double check)
                cand_action = bp_stmts[cand_idx]
                if not isinstance(cand_action, HIDE_ACTION):
                    print("merge 2")
                    cand_action.abs_state = merge_abs_state(
                        cand_action.abs_state, tmp_abs_state
                    )

                    # add else branch
                    new_if_else_branch = IFELSE(c_cond)
                    new_if_else_branch.stmts = bp_stmts[check_IF_id].stmts
                    new_if_else_branch.else_stmts = bp_stmts[check_IF_id + 1 : cand_idx]

                    # debug: try to add additional hidden action
                    new_if_else_branch.stmts.append(HIDE_ACTION(cand_action))
                    new_if_else_branch.else_stmts.append(HIDE_ACTION(cand_action))

                    bp_stmts[check_IF_id:cand_idx] = [new_if_else_branch]

                else:
                    # add else branch
                    print("add else branch")
                    new_if_else_branch = IFELSE(c_cond)
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
            if store_cost is None:
                cost, add_success = self.add_queue(
                    p, eval_reward, self.get_robot(self.seed)
                )
            else:
                cost, add_success = self.add_queue(
                    p, eval_reward, self.get_robot(self.seed), cost=store_cost - 1
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
                if iter == 392:
                    # pdb.set_trace()
                    pass
                try:
                    r, ts, p, robot, state, store_cost = self.q.get_nowait()
                except queue.Empty:
                    break
                # double check: debug
                # tmp_c_stmts, _ = p.find_actions(c_touch=True)
                # assert tmp_c_stmts is None
                # if iter == 24000 or iter == 26800:
                #     self.q = PriorityQueue()
                # log print
                log_and_print("searching base on {} with cost {}".format(str(p), r))

                # Execute Program
                if state is None:
                    # pdb.set_trace()
                    eval_result, eval_robot = self.eval_program(
                        self.seed, robot, p, check_multiple=False
                    )

                    # from matplotlib.pyplot import imsave; imsave("y.png", eval_robot.env.render(mode='rgb_array'))
                    eval_reward = eval_robot.check_reward()
                else:
                    eval_result, eval_reward = state
                    eval_robot = robot
                p.reset()

                # if str(p) == ' WHILE(not (markers_present)) { move} ; C ; END' or \
                #    str(p) == ' WHILE(not (markers_present)) { IF(not (front_is_clear)) { put_marker C }  move} ; C ; END' or \
                #    str(p) == ' WHILE(not (markers_present)) { IF(not (front_is_clear)) { put_marker turn_left C }  move} ; C ; END' or \
                #    str(p) == ' WHILE(not (markers_present)) { IF(not (front_is_clear)) { put_marker turn_left move C }  move} ; C ; END' or \
                #    str(p) == ' WHILE(not (markers_present)) { IF(not (front_is_clear)) { put_marker turn_left}  move} ; C ; END':
                #     if str(p) not in debug_store:
                #         debug_store[str(p)] = [copy.deepcopy(p), copy.deepcopy(eval_robot)]

                # get action before C
                c_stmts, c_idx = p.find_actions(c_touch=True)
                tmp_action = None
                if c_stmts is not None and len(c_stmts) > 1:
                    tmp_action = c_stmts[c_idx - 1]

                # 1) Success
                if eval_result == self.SUCCESS_TYPE:
                    # reset resume point
                    p.reset_resume_points()

                    c_stmts, c_idx = p.find_actions()
                    # TODO: double check
                    if c_stmts is not None and len(c_stmts) > 1:
                        tmp_action = c_stmts[c_idx - 1]

                    c_cond, cond_type = p.find_c_cond()
                    set_fail = False

                    # has C
                    if c_stmts is not None:
                        # get break point
                        bp_stmts, bp_idx = p.find_break_point()
                        if bp_stmts is not None:
                            bp = bp_stmts[bp_idx]
                            bp.break_point = False
                            if (
                                cond_type is not None
                                and cond_type == "i"
                                and str(tmp_action) == str(bp)
                            ):
                                # drop if
                                if len(c_stmts) == 1:
                                    bp_stmts.pop(bp_idx - 1)
                                # remove duplicate action
                                else:
                                    assert str(tmp_action) == str(c_stmts[c_idx - 1])
                                    c_stmts.pop(c_idx - 1)
                                print("merge 4")
                                bp.abs_state = merge_abs_state(
                                    bp.abs_state, tmp_action.abs_state
                                )

                        c_stmts, c_idx = p.find_actions()
                        # remove C
                        while c_stmts is not None:
                            if len(c_stmts) == 1:
                                # TODO: might be wiser to get into other seed
                                set_fail = True
                                break
                            c_stmts.pop(c_idx)
                            c_stmts, c_idx = p.find_actions()

                    if not set_fail:
                        p.reset()
                        eval_result, eval_robot = self.eval_program(
                            self.seed, self.get_robot(self.seed), p, check_multiple=True
                        )
                        if eval_result == self.MORE_WORK_TYPE:
                            eval_reward = eval_robot.check_reward()
                            log_and_print("more work from other seeds")
                            p.reset()
                        elif eval_result == self.SUCCESS_TYPE:
                            test_result = self.test_program(p)
                            if self.found_one and test_result == self.SUCCESS_TYPE:
                                with open("hook.txt", "w") as f:
                                    f.write(str(p))
                                break
                            else:
                                continue
                        else:
                            continue
                    else:
                        # check program empty
                        if iter == 0:
                            assert self.q.qsize() == 0
                            restart = True
                            self.seed = self.more_seeds[cur_id + 1]
                            self.q.put(
                                (
                                    0,  # reward
                                    time.time(),  # timestamp
                                    copy.deepcopy(self.sketch),
                                    self.get_robot(self.seed),
                                )
                            )
                            log_and_print(
                                "\nswitch from seed {} to {}".format(
                                    self.more_seeds[cur_id], self.more_seeds[cur_id + 1]
                                )
                            )
                            cur_id += 1
                            break
                        else:
                            log_and_print(
                                "\nfound but not success in all seeds (special) for \n {}".format(
                                    p
                                )
                            )
                            continue

                # 2) Fail
                elif eval_result == self.FAIL_TYPE:
                    continue

                # debug for MORE_WORK
                if p.check_resume_points():
                    outer_while_num = len(
                        [True for code in p.stmts if isinstance(code, WHILE)]
                    )
                    if not (
                        (isinstance(p.stmts[-2], C) and p.stmts[-2].touch)
                        or (outer_while_num > 1)
                    ):
                        pdb.set_trace()
                    assert (isinstance(p.stmts[-2], C) and p.stmts[-2].touch) or (
                        outer_while_num > 1
                    )
                    p.reset_resume_points()
                assert not p.check_resume_points()

                # 3) Find Break Point
                bp_stmts, bp_idx = p.find_break_point()
                # check whether if branch has been added
                expand_IF = False
                check_IF_id = None
                if bp_stmts is not None:
                    bp = bp_stmts[bp_idx]
                    for check_id in range(bp_idx - 1, -1, -1):
                        if isinstance(bp_stmts[check_id], IF):
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
                        assert c_stmts is None
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
                    if isinstance(code, ACTION):
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
                        return

                # Expand (drop when while;C)
                # set resume point
                p.set_resume_points()

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
                            cur_abs_state = get_abs_state(new_robot)
                            prev_abs_state = get_abs_state(eval_robot)
                            new_reward = new_robot.check_reward()
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
                                c_stmts, c_idx = candidate.find_actions(c_touch=False)
                                c_stmts.pop(c_idx)

                    candidate.reset_c_touch()
                    # debug
                    c_stmts, c_idx = candidate.find_actions(c_touch=True)
                    assert c_stmts is None
                    # add back
                    new_robot = self.copy_robot(eval_robot)
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
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="store", default="search")
    parser.add_argument("--make_video", action="store_const", const=True, default=False)
    parser.add_argument("--seed", action="store", required=True)
    parser.add_argument("--random_seed", action="store", required=True)
    args = parser.parse_args()
    random.seed(int(args.random_seed))
    np.random.seed(int(args.random_seed))

    df = pd.DataFrame([], columns=['Step', 'Value'])
    # NOTE: for simplicity, not a tree right now
    program_db = []

    seed = int(args.seed)
    more_seeds = [i for i in range(10, 15)]
    eval_seeds = [i for i in range(1, 1500)]

    program = Program().expand()[0].expand()[3].expand()[0].expand()[0]
    program.stmts[0].stmts.append(
        get_action(
            "idle",
            get_abs_state_from_list(
                [
                    "DNC",
                    "DNC",
                    "DNC",
                    "DNC",
                    "DNC",
                    "DNC",
                    "DNC",
                ]
            ),
            get_abs_state_from_list(
                [
                    "DNC",
                    "DNC",
                    "DNC",
                    "DNC",
                    "DNC",
                    "DNC",
                    "DNC",
                ]
            ),
        )
    )
    print("starting from program ", program)
    node = Node(
        sketch=program,
        task="FetchHookOptimized-v0",
        seed=seed,
        more_seeds=more_seeds,
        eval_seeds=eval_seeds,
        max_search_iter=10000,
        max_structural_cost=20,
        shuffle_actions=True,
        found_one=True,
    )

    if args.make_video:
        # remove old files
        files = os.listdir("frames/")
        for f in files:
            os.remove(os.path.join("frames", f))

        dsl.DSL_DEBUG = True

    if args.task == "search":
        node.search()
        df.to_csv(f"hook-final-{args.seed}-{args.random_seed}_random_0.2_100_steps.csv")
    elif args.task == "test":
        rate = node.test_for_success_rate(program)
        from termcolor import colored

        print(colored(f"success rate is {rate * 100}%", "light_yellow"))

    if args.make_video:
        os.system("bash make_video.sh")
