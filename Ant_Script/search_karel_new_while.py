# NOTE: each node is associated with a sketch
import enum
import queue
import time
import copy
from queue import PriorityQueue
import multiprocessing as mp

import numpy as np
import gym
from multiworld.envs.mujoco import register_custom_envs as register_mujoco_envs

from configs import get_configs
from dsl import *
from ant_program_env import *

from utils.logging import log_and_print

import pdb

# all primitive policies
ANT_LOW_TORQUE_MODELS = []
for direction in ['up', 'down', 'left', 'right']:
    filename = os.getcwd() + '/primitives/ant_low_torque/' + direction + '.pt'
    ANT_LOW_TORQUE_MODELS.append(torch.load(filename))

start_time = None

# def get_structural_cost(program):
#     cost = 0
#     program_str = str(program)
#     for s in ACTION_DICT:
#         cost += program_str.count(s)
#     # debug try to also calculate if
#     # cost += program_str.count('IF')

#     return cost

def get_structural_cost(program):
    cost = 0
    program_str = str(program)
    for s in ACTION_DICT:
        cost += program_str.count(s)
    # debug try to also calculate if
    cost += program_str.count('IF')

    return cost


def execute_for_reward(cur_robot, program, force=True, steps=False):
    cur_prog = copy.deepcopy(program)
    cur_prog.reset()
    # cur_prog.reset_resume_points()
    cur_robot.force_execution = force
    
    cur_prog.execute(cur_robot)

    if steps:
        return cur_robot.no_fuel(), cur_robot.check_reward(), cur_prog.complete(), cur_robot.steps
    else:
        return cur_robot.no_fuel(), cur_robot.check_reward(), cur_prog.complete()

def get_specific_if(cond):
    new_if = IF(cond=copy.deepcopy(cond))
    # C[0]
    new_if.stmts[0] = C(unit=True)
    # WHILE(cond){C}
    new_while = WHILE()
    new_while.cond = [copy.deepcopy(cond)]
    new_while.stmts[0] = C()
    new_if.stmts.append(new_while)
    # C
    new_if.stmts.append(C())

    return new_if

def copy_robot(robot):
    new_robot = copy.deepcopy(robot)
    trans_state = robot.env.env.get_env_state()
    new_robot.env.env.set_env_state(trans_state)

    return new_robot

# NOTE: each node is associated with a sketch
class Node:
    # constant
    SUCCESS_TYPE = 'success'
    NEXT_GOAL_TYPE = 'next_goal'
    FAIL_TYPE = 'fail'
    MORE_WORK_TYPE = 'more_work'

    # init
    def __init__(self, 
            sketch, 
            task, 
            seed=123,
            more_seeds=[],
            eval_seeds = [],
            max_search_iter = 50,
            max_structural_cost = 30,
            shuffle_actions=False,
            found_one=False,
            prob_mode=False,
            sub_goals=[1.0],
            multi_eval=True,
            more_tasks=[]):
        
        self.sketch = sketch
        self.task = task
        self.seed = seed
        self.more_seeds = more_seeds
        self.eval_seeds = eval_seeds
        self.max_search_iter = max_search_iter
        self.max_structural_cost = max_structural_cost
        self.shuffle_actions = shuffle_actions
        self.found_one = found_one
        self.multi_eval = multi_eval

        # store all required robot
        config_data = get_configs(self.task)
        if task == 'AntMaze':
            config_dict['max_episode_length'] = 1000
        config_data['distance_threshold']['front'] = 1.2
        self.config_data = config_data

        # store all required robot
        new_robot = AntProgramEnv(
                        env=self.get_env(self.task),
                        models=ANT_LOW_TORQUE_MODELS,
                        goal_threshold=0.5,
                        goal_detection_threshold=4.0,
                        seed=seed,
                        **config_data)
        new_robot.reset()
        self.robot_store = {self.seed: new_robot}

        for e_id, e in enumerate(more_seeds):
            if len(more_tasks) >= len(more_seeds):
                cur_task = more_tasks[e_id]
            else:
                cur_task = self.task

            new_robot = AntProgramEnv(
                            env=self.get_env(self.task),
                            models=ANT_LOW_TORQUE_MODELS,
                            goal_threshold=0.5,
                            goal_detection_threshold=4.0,
                            seed=e,
                            **config_data)
            new_robot.reset()
            self.robot_store[e] = new_robot
        for e in eval_seeds:
            new_robot = AntProgramEnv(
                            env=self.get_env(self.task),
                            models=ANT_LOW_TORQUE_MODELS,
                            goal_threshold=0.5,
                            goal_detection_threshold=4.0,
                            seed=e,
                            **config_data)
            new_robot.reset()
            self.robot_store[e] = new_robot

        # init sub goals
        self.reject_all = False
        self.sub_goals = sub_goals
        self.cur_goal_idx = 0
        for seed in self.robot_store:
            self.robot_store[seed].cur_goal = self.sub_goals[self.cur_goal_idx]

        self.q = PriorityQueue()
        self.q.put((
            0,  # reward
            time.time(), # timestamp
            copy.deepcopy(self.sketch),
            self.get_robot(self.seed),
            None, # last evaluation
            None, # cost
        ))
        # NOTE: tmp code, store potential case 3 programs? not so sure
        self.case_3_list = []
 
        self.candidates = {  # store success (and unsuccess) programs
            'success': [],
            'success_search': [],
            'failed': [],
            'no_fuel': [],
            'complete': []
        }

        # TODO: # number of top-k programs we select from a node
        self.k = 1

        # multi process
        self.pool = mp.Pool(processes=len(self.more_seeds) + 1)

        # debug
        self.timesteps = [0]
        self.all_rewards = []
        self.eval_rewards = []

    # get new env
    def get_env(self, task):
        if self.task == 'AntU':
            env = gym.make('AntULongTestEnv-v0', disable_env_checker=True)
        elif self.task == 'AntFb':
            env = gym.make('AntFbMedTestEnv-v1', disable_env_checker=True)
        elif self.task == 'AntMaze':
            env = gym.make('AntMazeMedTestEnv-v1', disable_env_checker=True)
        elif self.task == 'AntFg':
            env = gym.make('AntFgMedTestEnv-v1', disable_env_checker=True)
        return env
    
    # get robot from store
    def get_robot(self, seed):
        robot = copy_robot(self.robot_store[seed])

        # robot = AntProgramEnv(
        #                 env=self.get_env(self.task),
        #                 models=ANT_LOW_TORQUE_MODELS,
        #                 goal_threshold=0.5,
        #                 goal_detection_threshold=4.0,
        #                 seed=seed,
        #                 **self.config_data)
        # robot.cur_goal = self.sub_goals[self.cur_goal_idx]
        # robot.reset()

        return robot

    # add candidate program into queue
    def add_queue(self, candidate, reward, robot, cost=None):
        global start_time
        log_and_print('current time: {}'.format(time.time() - start_time))

        # calculate reward before add
        if self.multi_eval:
            attempt_id = 0
            while True:
                eval_result, robot, reward, eval_reward = self.eval_program(self.seed, robot, candidate, check_multiple=True)
                if eval_result != self.MORE_WORK_TYPE:
                    break
                elif robot.check_reward() < self.sub_goals[self.cur_goal_idx]:
                    break
                pdb.set_trace()
                attempt_id += 1
                if attempt_id >= 100:
                    pdb.set_trace()
    
        else:
            eval_result, robot, reward, eval_reward = self.single_eval_program(self.seed, robot, candidate, check_multiple=True)
        # reward = robot.check_reward()
        # reward = robot.check_eval_reward()

        # add back
        # assert cost is None
        if cost is None:
            cost = get_structural_cost(candidate)

        # clear queue and next goal
        if eval_result == Node.NEXT_GOAL_TYPE:
            if self.reject_all == False:
                self.q = PriorityQueue()
                self.reject_all = True

        elif self.reject_all or eval_result != Node.MORE_WORK_TYPE:
            if self.reject_all:
                log_and_print('reject all')
            return cost, reward, False

        q_tuple = (
            # -reward + cost * 0.02,
            -reward + cost * 0.04,
            # -reward + cost * 0.06,
            # -reward + cost * 0.1,
            # -reward + cost * 0.15,
            # -reward + cost * 0.2,
            # -reward + cost * 0.3,
            time.time(),
            candidate,
            robot,
            (eval_result, reward, eval_reward),
            None,
        )
        if not cost > self.max_structural_cost:
            self.q.put(q_tuple)

            return cost, reward, True
        return cost, reward, False

    # check whether C exist in statement
    def check_C_exist(self, stmts, touch=False):
        exist = False
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, HIDE_ACTION, ant_cond, END)):
                pass
            elif isinstance(code, (WHILE, IF, IFELSE)):
                exist = self.check_C_exist(code.stmts, touch=touch)
                if exist:
                    return exist
            elif isinstance(code, C):
                if touch:
                    if code.touch:
                        return True
                else:
                    return True
            else:
                raise ValueError('Invalide code')
        
        return exist

    def test_program(self, prog):
        for e in self.eval_seeds:
            # force evaluate
            force_eval_robot = self.get_robot(e)
            force_eval_robot.force_execution = True
            prog.execute(force_eval_robot)
            prog.reset()
            force_eval_robot.force_execution = False
            passed = force_eval_robot.check_reward() == 1
            # fail
            if not passed:
                # log and print
                log_and_print('\nfound but not success in all seeds for \n {}'.format(prog))
                self.candidates['success_search'].append((1, prog))
                return self.FAIL_TYPE
        # success
        log_and_print('\nsuccess and store for \n {}'.format(prog))
        self.candidates['success'].append((1, prog))
        return self.SUCCESS_TYPE

    def check_subgoal_valid(self, candidate, target_goal):
        # init
        all_seeds = [self.seed] + self.more_seeds
        all_robots = [self.get_robot(e) for e in all_seeds]
        for robot in all_robots:
            robot.cur_goal = target_goal

        # init again
        all_rewards = []
        all_no_fuel = []
        all_completes = []

        # prepare processes
        results = [self.pool.apply_async(execute_for_reward, args=(robot, candidate, False)) \
                       for robot in all_robots]
        results = [p.get() for p in results]

        # store
        for robot_no_fuel, reward, complete in results:
            # C or breakpoint
            all_rewards.append(reward)
            all_no_fuel.append(robot_no_fuel)
            all_completes.append(complete)

        # test
        for seed, robot_no_fuel, reward, complete in zip(all_seeds, all_no_fuel, all_rewards, all_completes):
            if reward == -1:
                return False
            elif reward < target_goal and robot_no_fuel:
                return False
            elif reward < target_goal and complete:
                return False
        
        return True


    def eval_program(self, seed, robot, candidate, check_multiple):
        # init
        single_seed = len(self.more_seeds) == 0
        check_multiple = False if single_seed else check_multiple
        if self.sub_goals is None:
            cur_goal = 1
        else:
            cur_goal = self.sub_goals[self.cur_goal_idx]

        # execute and get reward
        # eval_robot = self.get_robot(seed)
        eval_robot = robot
        
        before_steps = eval_robot.steps
        candidate.execute(eval_robot)
        self.timesteps[-1] += eval_robot.steps - before_steps

        r = eval_robot.check_reward()
        complete = candidate.complete()
        seed = eval_robot.seed

        # if 'WHILE(not (present_goal)) { IF(not (front_is_clear)) { turn_left WHILE(not (front_is_clear)) { C } ; move C }  move} ; C ; END' in str(candidate):
        #     pdb.set_trace()

        # fail cases
        # fail
        if r == -1:
            # log and print
            log_and_print('\n fail for \n {}'.format(candidate))
            self.candidates['failed'].append((-1, candidate))
            return self.FAIL_TYPE, eval_robot, -1, -1

        # success or more work
        all_rewards = [r]
        all_no_fuel = [eval_robot.no_fuel()]
        all_completes = [complete]
        # success_seeds = [seed]
        all_seeds = [self.seed] + self.more_seeds
        all_seeds.pop(all_seeds.index(seed))

        # prepare processes
        # pdb.set_trace()
        # results = [self.pool.apply_async(execute_for_reward, args=(self.get_robot(e), candidate, False, True)) \
        #                for e in all_seeds]
        # results = [p.get() for p in results]
        results = []
        for e in all_seeds:
            results.append(execute_for_reward(self.get_robot(e), candidate, False, True))

        # if 'WHILE(not (present_goal)) { IF(not (front_is_clear)) { turn_left WHILE(not (front_is_clear)) { move C } ;}  move} ; C ; END' in str(candidate):
        #     pdb.set_trace()

        for robot_no_fuel, reward, complete, timesteps in results:
            # C or breakpoint
            all_rewards.append(reward)
            all_no_fuel.append(robot_no_fuel)
            all_completes.append(complete)
            self.timesteps[-1] += timesteps

        # assert np.mean(all_rewards) == r
        log_and_print('all rewards: {}'.format(all_rewards))

        # calculate evaluation reward
        if len(self.eval_seeds) > 0:
            # eval_results = [self.pool.apply_async(execute_for_reward, args=(self.get_robot(e), candidate, True, True)) \
            #             for e in self.eval_seeds]
            # eval_results = [p.get()[1] for p in eval_results]
            eval_results = []
            for e in self.eval_seeds:
                eval_results.append(execute_for_reward(self.get_robot(e), candidate, True, True)[1])
            eval_reward = np.mean(eval_results)
        else:
            eval_reward = 0

        # check fail
        all_seeds = [seed] + all_seeds
        for seed, robot_no_fuel, reward, complete in zip(all_seeds, all_no_fuel, all_rewards, all_completes):
            if reward == -1:
                # log and print
                log_and_print('\n fail for \n {}'.format(candidate))
                self.candidates['failed'].append((-1, candidate))
                return self.FAIL_TYPE, eval_robot, -1, eval_reward
            elif reward < cur_goal and robot_no_fuel:
                # log and print
                log_and_print('\n no fuel with reward {} under seed {} for \n {}'.format(reward, seed, candidate))
                self.candidates['no_fuel'].append((r, candidate))
                return self.FAIL_TYPE, eval_robot, r, eval_reward
            elif reward < cur_goal and complete:
                # log and print
                log_and_print('\n complete with reward {} under seed {} for\n {}'.format(reward, seed, candidate))
                self.candidates['complete'].append((r, candidate))
                return self.FAIL_TYPE, eval_robot, r, eval_reward

        # more work on search seed
        if r < cur_goal:
            if candidate.count_C() == 1 and isinstance(candidate.stmts[-2], C):
                log_and_print('\n special complete with reward {} for\n {}'.format(np.mean(all_rewards), candidate))
                self.candidates['complete'].append((np.mean(all_rewards), candidate))
            
            return self.MORE_WORK_TYPE, eval_robot, np.mean(all_rewards), eval_reward

        # check success
        if np.mean(np.array(all_rewards) >= cur_goal) == 1:
            # log and print
            if self.cur_goal_idx == len(self.sub_goals) - 1:
                log_and_print('\n success and store for {}'.format(candidate))
                self.candidates['success'].append((cur_goal, candidate))

                # add logs
                self.all_rewards.append(cur_goal)
                self.eval_rewards.append(eval_reward)

                return self.SUCCESS_TYPE, eval_robot, cur_goal, eval_reward
            
            else:
                log_and_print('\n success for current goal {} for {}'.format(cur_goal, candidate))

                # test
                valid = self.check_subgoal_valid(copy.deepcopy(candidate), self.sub_goals[self.cur_goal_idx+1])
                if valid:
                    # get back to search seed
                    candidate.reset()
                    eval_robot = self.get_robot(eval_robot.seed)
                    eval_robot.cur_goal = self.sub_goals[self.cur_goal_idx+1]
                    candidate.execute(eval_robot)

                    return self.NEXT_GOAL_TYPE, eval_robot, np.mean(all_rewards), eval_reward
                else:
                    log_and_print('invalid for next goal {}'.format(self.sub_goals[self.cur_goal_idx+1]))
                    return self.FAIL_TYPE, eval_robot, np.mean(all_rewards), eval_reward

        else:
            log_and_print('\nfound but not success in all seeds with reward {} for \n {}'.format(np.mean(all_rewards), candidate))
            self.candidates['success_search'].append((np.mean(all_rewards), candidate))
            
            for seed, reward in zip(all_seeds, all_rewards):
                if reward < cur_goal:
                    new_seed = seed

            candidate.reset()
            eval_robot = self.get_robot(new_seed)
            candidate.execute(eval_robot)

            # if 'WHILE(not (present_goal)) { IF(not (front_is_clear)) { turn_left WHILE(not (front_is_clear)) { move C } ;}  move} ; C ; END' in str(candidate):
            #     pdb.set_trace()

            log_and_print('switch to robot seed {}'.format(new_seed))

            return self.MORE_WORK_TYPE, eval_robot, np.mean(all_rewards), eval_reward
        

    def single_eval_program(self, seed, robot, candidate, check_multiple):
        # init
        single_seed = len(self.more_seeds) == 0
        check_multiple = False if single_seed else check_multiple
        if self.sub_goals is None:
            cur_goal = 1
        else:
            cur_goal = self.sub_goals[self.cur_goal_idx]

        # execute and get reward
        # eval_robot = self.get_robot(seed)
        eval_robot = robot
        
        before_steps = eval_robot.steps
        candidate.execute(eval_robot)
        self.timesteps[-1] += eval_robot.steps - before_steps

        r = eval_robot.check_reward()
        complete = candidate.complete()
        seed = eval_robot.seed

        # calculate evaluation reward
        if len(self.eval_seeds) > 0:
            eval_results = [self.pool.apply_async(execute_for_reward, args=(self.get_robot(e), candidate, True, True)) \
                        for e in self.eval_seeds]
            eval_results = [p.get()[1] for p in eval_results]
            eval_reward = np.mean(eval_results)
        else:
            eval_reward = 0

        # fail cases
        # fail
        if r == -1:
            # log and print
            log_and_print('\n fail for \n {}'.format(candidate))
            self.candidates['failed'].append((-1, candidate))
            return self.FAIL_TYPE, eval_robot, -1, -1
        elif r < cur_goal:
            if eval_robot.no_fuel():
                # log and print
                log_and_print('\n no fuel with reward {} under seed {} for \n {}'.format(r, seed, candidate))
                self.candidates['no_fuel'].append((r, candidate))
                return self.FAIL_TYPE, eval_robot, r, eval_reward
            elif complete:
                # log and print
                log_and_print('\n complete with reward {} under seed {} for\n {}'.format(r, seed, candidate))
                self.candidates['complete'].append((r, candidate))
                return self.FAIL_TYPE, eval_robot, r, eval_reward
            # more work on search seed
            else:
                if candidate.count_C() == 1 and isinstance(candidate.stmts[-2], C):
                    log_and_print('\n special complete with reward {} for\n {}'.format(r, candidate))
                    self.candidates['complete'].append((r, candidate))
                
                return self.MORE_WORK_TYPE, eval_robot, r, eval_reward
        else:
            all_seeds = [self.seed] + self.more_seeds
            all_seeds.pop(all_seeds.index(seed))
            # check success
            for e in all_seeds:
                new_robot = self.get_robot(e)
                candidate.reset()
                candidate.force_execution = False
                candidate.execute(new_robot)
                e_no_fuel, e_r, e_complete, e_steps = new_robot.no_fuel(), new_robot.check_reward(), candidate.complete(), new_robot.steps

                self.timesteps[-1] += e_steps
                if e_r < cur_goal:
                    if e_r == -1:
                        # log and print
                        log_and_print('\n fail for seed {} \n {}'.format(e, candidate))
                        self.candidates['failed'].append((-1, candidate))
                        return self.FAIL_TYPE, new_robot, -1, -1
                    elif e_no_fuel:
                        # log and print
                        log_and_print('\n no fuel with reward {} under seed {} for \n {}'.format(e_r, e, candidate))
                        self.candidates['no_fuel'].append((e_r, candidate))
                        return self.FAIL_TYPE, new_robot, e_r, eval_reward
                    elif e_complete:
                        # log and print
                        log_and_print('\n complete with reward {} under seed {} for\n {}'.format(e_r, seed, candidate))
                        self.candidates['complete'].append((e_r, candidate))
                        return self.FAIL_TYPE, new_robot, e_r, eval_reward
                    else:
                        log_and_print('\nfound but not success in all seeds with reward {} for \n {}'.format(e_r, candidate))
                        log_and_print('switch to robot seed {}'.format(e))

                        return self.MORE_WORK_TYPE, new_robot, e_r, eval_reward
            # success
            if self.cur_goal_idx == len(self.sub_goals) - 1:
                log_and_print('\n success and store for {}'.format(candidate))
                self.candidates['success'].append((cur_goal, candidate))

                # add logs
                self.all_rewards.append(cur_goal)
                self.eval_rewards.append(eval_reward)

                return self.SUCCESS_TYPE, eval_robot, cur_goal, eval_reward
            # next goal
            else:
                print('need further debug')
                pdb.set_trace()
                eval_robot = self.get_robot(eval_robot.seed)
                eval_robot.cur_goal = self.sub_goals[self.cur_goal_idx+1]
                candidate.reset()
                candidate.execute(eval_robot)

                return self.NEXT_GOAL_TYPE, eval_robot, r, eval_reward


    # Add IF (case 1 | expand for one action) or Case 3
    def add_if_branch(self, candidate, eval_reward, eval_robot, store_cost=None):
        robot_seed = eval_robot.seed
        # check break point
        bp_stmts, bp_idx = candidate.find_break_point()
        if bp_stmts is None:
            return

        # if 'WHILE(not (present_goal)) { IF(not (front_is_clear)) { turn_left WHILE(not (front_is_clear)) { C } ; C }  move} ; C ; END' in str(candidate):
        #     pdb.set_trace()

        # get condition
        bp = bp_stmts[bp_idx]
        diff_conds = get_diff_conds(bp.abs_state, bp.obs_abs_state)

        # find starting id of IFS before bp
        loc = bp_idx
        while loc-1 >= 0 and isinstance(bp_stmts[loc-1], IF):
            loc -= 1
        if_start_id = loc

        # add if branch (case 1 | expand for one action)
        case_1_applied = False
        for j in range(len(diff_conds)):
            # insert IF(cond) {C} at break point
            for k in range(if_start_id, bp_idx+1):
                # add IF
                # new_if = IF(cond=diff_conds[j])
                new_if = get_specific_if(diff_conds[j])
                new_if.stmts[0].touch = True
                bp_stmts.insert(k, new_if)
                tmp_cand = copy.deepcopy(candidate)
                bp_stmts.pop(k)
                
                # add action
                new_cand_list, new_action_list = tmp_cand.expand_actions(c_touch=True)
                for new_cand, new_action in zip(new_cand_list, new_action_list):
                    # case one (only once)
                    if str(new_action) == str(bp):
                        if case_1_applied:
                            continue

                        # update abstract state
                        prev_abs_state = bp.abs_state
                        prev_obs_abs_state = bp.obs_abs_state
                        new_abs_state = merge_abs_state(prev_abs_state, bp.obs_abs_state)

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

                        # debug get new robot
                        new_robot = self.get_robot(robot_seed)
                        if store_cost is None:
                            cost, new_reward, add_success = self.add_queue(new_cand, eval_reward, new_robot)
                        else:
                            cost, new_reward, add_success = self.add_queue(new_cand, eval_reward, new_robot, cost=store_cost+1)

                        # log and print
                        if add_success:
                            log_and_print('case one applied: put into queue and get cost {} reward {} for seed {} for \n {}'.format(cost, new_reward, robot_seed, str(new_cand)))
                        else:
                            log_and_print('unable to put into queue and get cost {} reward {} for seed {} for\n {}'.format(cost, new_reward, robot_seed, str(new_cand)))
                        continue

                    # direct add
                    new_cand.reset_c_touch()
                    # debug
                    tmp_c_stmts, tmp_c_idx = new_cand.find_actions(c_touch=True)
                    assert tmp_c_stmts is None

                    # debug get new robot
                    new_robot = self.get_robot(robot_seed)
                    if store_cost is None:
                        cost, new_reward, add_success = self.add_queue(new_cand, eval_reward, new_robot)
                    else:
                        cost, new_reward, add_success = self.add_queue(new_cand, eval_reward, new_robot, cost=store_cost+1)

                    # log and print
                    if add_success:
                        log_and_print('add if: put into queue and get cost {} reward {} for seed {} for\n {}'.format(cost, new_reward, robot_seed, str(new_cand)))
                    else:
                        log_and_print('unable to put into queue and get cost {} reward {} for seed {} for\n {}'.format(cost, new_reward, robot_seed, str(new_cand)))
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
                if not isinstance(tmp_end_action, ACTION) or tmp_end_action.post_abs_state is None:
                    continue

                # attempt to execution a single action
                # print('double check state of evaluate robot')
                # pdb.set_trace()
                # _eval_robot = copy.deepcopy(eval_robot)
                _eval_robot = copy_robot(eval_robot)
                _eval_robot.execute_single_action(tmp_end_action.action)
                _eval_robot_current_state = get_abs_state(_eval_robot)

                if satisfy_abs_state(_eval_robot_current_state, tmp_end_action.post_abs_state):
                    for j in range(len(diff_conds)):
                        # clone
                        _candidate = copy.deepcopy(candidate)
                        _bp_stmts, _bp_idx = _candidate.find_break_point()

                        neg_cond = get_neg_cond(diff_conds[j])
                        IF_code = IF(cond=neg_cond)
                        IF_code.stmts.pop()  # remove C
                        IF_code.stmts += _bp_stmts[start_idx: end_idx]
                        # debug: try to add additional hidden action
                        IF_code.stmts.append(HIDE_ACTION(_bp_stmts[end_idx]))

                        start_end_idx.append((start_idx, end_idx))
                        new_IF_code.append(IF_code)
                        new_candidate_code.append(_candidate)

            # consider special case when inside if branch (can combine with above case just specify here to mark special case)
            if isinstance(bp_stmts[-1], HIDE_ACTION):
                # pdb.set_trace()
                assert len_bp_stmts - bp_idx > 1

                # if str(candidate) == ' WHILE(all_true) { IF(not (right_is_clear)) { IF(not (front_is_clear)) { lane_left} ELSE { faster} idle} ELSE { lane_right} idle} ;; END':
                #     pdb.set_trace()

                # _eval_robot = copy.deepcopy(eval_robot)
                _eval_robot = copy_robot(eval_robot)
                _eval_robot.execute_single_action(bp_stmts[-1].action.action)
                _eval_robot_current_state = get_abs_state(_eval_robot)

                if satisfy_abs_state(_eval_robot_current_state, bp_stmts[-1].action.post_abs_state):
                    for j in range(len(diff_conds)):
                        # clone
                        _candidate = copy.deepcopy(candidate)
                        _bp_stmts, _bp_idx = _candidate.find_break_point()

                        neg_cond = get_neg_cond(diff_conds[j])
                        IF_code = IF(cond=neg_cond)
                        IF_code.stmts.pop()  # remove C
                        IF_code.stmts += _bp_stmts[start_idx:]

                        start_end_idx.append((start_idx, len_bp_stmts-1))
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
            _bp_stmts[_start: _end] = []
            _bp_stmts.insert(_start, new_IF_code[k])
            _candidate.reset()

            # add to queue
            new_robot = self.get_robot(robot_seed)
            if store_cost is None:
                cost, new_reward, add_success = self.add_queue(_candidate, eval_reward, new_robot)
            else:
                cost, new_reward, add_success = self.add_queue(_candidate, eval_reward, new_robot, cost=store_cost+1)

            # log print
            if add_success:
                log_and_print('case 3: put into queue and get cost {} reward {} for seed {} for\n {}'.format(cost, new_reward, robot_seed, str(_candidate)))
            else:
                log_and_print('unable to put into queue and get cost {} reward {} for seed {} for\n {}'.format(cost, new_reward, robot_seed, str(_candidate)))


    def case_two_drop(self, p, tmp_action, tmp_abs_state, tmp_pos_abs_state, bp_idx, bp_stmts, c_idx, c_cond, c_stmts, \
                      eval_reward, check_IF_id, eval_robot, robot_seed, store_cost=None):
        bp = bp_stmts[bp_idx]

        # if ' WHILE(all_true) { IF(not (right_is_clear)) { faster idle C }  lane_right idle} ;; END' == str(p):
        #     pdb.set_trace()

        # if 'WHILE(not (present_goal)) { IF(not (front_is_clear)) { turn_left WHILE(not (front_is_clear)) { C } ; move C }  move} ; C ; END' in str(p):
        #     pdb.set_trace()

        # find all satisfied actions
        cand_idx = None
        for idx in range(bp_idx, len(bp_stmts)):
            cand_action = bp_stmts[idx]
            abs_state_check = (isinstance(cand_action, ACTION) and satisfy_abs_state(tmp_pos_abs_state, cand_action.post_abs_state)) or \
                (isinstance(cand_action, HIDE_ACTION) and satisfy_abs_state(tmp_pos_abs_state, cand_action.action.post_abs_state))
            action_check = (isinstance(cand_action, ACTION) and str(tmp_action) == str(cand_action)) or \
                            (isinstance(cand_action, HIDE_ACTION) and str(tmp_action) == str(cand_action.action))
            if action_check and abs_state_check:
                cand_idx = idx
                break

        # end if or add else
        if cand_idx is not None:
            # debug
            # assert bp_stmts[check_IF_id].stmts == c_stmts

            bp.break_point = False
            # remove C
            c_stmts.pop(c_idx)
            # remove duplicate action
            c_stmts.pop(c_idx-1)
            assert len(c_stmts) == c_idx-1

            # directly end if
            if cand_idx == bp_idx:
                # update abstract state
                bp.abs_state = merge_abs_state(bp.abs_state, tmp_abs_state)
                bp.abs_state.dnc_cond(str(c_cond.cond))
                
                # add hidden action
                hidden_action = HIDE_ACTION(bp)
                c_stmts.append(hidden_action)

            # add else branch
            else:
                # update abstract state (double check)
                cand_action = bp_stmts[cand_idx]
                if not isinstance(cand_action, HIDE_ACTION):
                    cand_action.abs_state = merge_abs_state(cand_action.abs_state, tmp_abs_state)
                    cand_action.abs_state.dnc_cond(str(c_cond.cond))

                    # add else branch
                    new_if_else_branch = IFELSE(c_cond)
                    new_if_else_branch.stmts = bp_stmts[check_IF_id].stmts
                    new_if_else_branch.else_stmts = bp_stmts[check_IF_id+1:cand_idx]

                    # debug: try to add additional hidden action
                    new_if_else_branch.stmts.append(HIDE_ACTION(cand_action))
                    new_if_else_branch.else_stmts.append(HIDE_ACTION(cand_action))

                    bp_stmts[check_IF_id:cand_idx] = [new_if_else_branch]

                else:
                    # add else branch
                    new_if_else_branch = IFELSE(c_cond)
                    new_if_else_branch.stmts = bp_stmts[check_IF_id].stmts
                    # debug: try to add additional hidden action
                    new_if_else_branch.stmts.append(cand_action)
                    new_if_else_branch.else_stmts = bp_stmts[check_IF_id+1:]

                    bp_stmts[check_IF_id:cand_idx] = [new_if_else_branch]

            # add back
            p.reset_c_touch()
            p.reset_resume_points()
            # debug
            c_stmts, c_idx = p.find_actions(c_touch=True)
            assert c_stmts is None

            new_robot = self.get_robot(robot_seed)
            if store_cost is None:
                cost, new_reward, add_success = self.add_queue(p, eval_reward, new_robot)
            else:
                cost, new_reward, add_success = self.add_queue(p, eval_reward, new_robot, cost=store_cost+1)

            # log print
            if add_success:
                log_and_print('case 2: put into queue and get cost {} reward {} for seed {} for\n {}'.format(cost, new_reward, robot_seed, str(p)))
            else:
                log_and_print('unable to put into queue and get cost {} reward {} for seed {} for\n {}'.format(cost, new_reward, robot_seed, str(p)))

        return cand_idx


    # Note:
    # Execute Program:
    # -> Success -> multi. seed -> END
    # -> Fail -> END
    # -> Break Point -> Add IF (case 1 | expand for one action) or Case 3  (should before Find C)
    # -> Find C -> Delete (case 2 | while finish) or Expand (while | IF | drop C (while;C))
    def search(self):
        # debug_store = {}
        global start_time
        start_time = time.time()

        cur_id = 0
        while True:
            restart = False
            # do search
            for iter in range(self.max_search_iter):
                log_and_print('[ITER] {}'.format(iter))
                if self.reject_all:
                    self.cur_goal_idx += 1
                    for seed in self.robot_store:
                        self.robot_store[seed].cur_goal = self.sub_goals[self.cur_goal_idx]
                    self.reject_all = False

                if self.found_one and len(self.candidates['success']) > 0:
                    break

                # get one program
                try:
                    r, ts, p, robot, state, store_cost = self.q.get_nowait()
                except queue.Empty:
                    break

                # double check: debug
                # tmp_c_stmts, _ = p.find_actions(c_touch=True)
                # assert tmp_c_stmts is None

                # log print
                log_and_print('searching base on {} with cost {}'.format(str(p), r))

                # Execute Program
                if state is None:
                    attempt_id = 0
                    while True:
                        if self.multi_eval:
                            eval_result, eval_robot, eval_reward, test_reward = self.eval_program(self.seed, robot, p, check_multiple=False)
                        else:
                            eval_result, eval_robot, eval_reward, test_reward = self.single_eval_program(self.seed, robot, p, check_multiple=False)
                        if eval_result != self.MORE_WORK_TYPE:
                            break
                        elif eval_robot.check_reward() < self.sub_goals[self.cur_goal_idx]:
                            break
                        pdb.set_trace()
                        attempt_id += 1
                        if attempt_id >= 100:
                            pdb.set_trace()

                else:
                    eval_result, eval_reward, test_reward = state
                    eval_robot = robot
                robot_seed = eval_robot.seed
                p.reset()

                # debug 
                self.all_rewards.append(eval_reward)
                self.eval_rewards.append(test_reward)
                self.timesteps.append(self.timesteps[-1])

                # get action before C
                c_stmts, c_idx = p.find_actions(c_touch=True)
                tmp_action = None
                if c_stmts is not None and len(c_stmts)>1:
                    # pdb.set_trace()
                    tmp_action = c_stmts[c_idx-1]

                # 1) Success
                if eval_result == self.SUCCESS_TYPE:
                    # should not get here
                    break

                # 2) Fail
                elif eval_result == self.FAIL_TYPE:
                    pdb.set_trace()

                # debug for MORE_WORK
                if p.check_resume_points():
                    outer_while_num = len([True for code in p.stmts if isinstance(code, WHILE)])
                    if not ((isinstance(p.stmts[-2], C) and p.stmts[-2].touch) or (outer_while_num>1)):
                        pdb.set_trace()
                    assert (isinstance(p.stmts[-2], C) and p.stmts[-2].touch) or (outer_while_num>1)
                    p.reset_resume_points()
                assert not p.check_resume_points()

                # 3) Find Break Point
                bp_stmts, bp_idx = p.find_break_point()
                # check whether if branch has been added
                expand_IF = False
                check_IF_id = None
                if bp_stmts is not None:
                    bp = bp_stmts[bp_idx]
                    for check_id in range(bp_idx-1, -1, -1):
                        if isinstance(bp_stmts[check_id], IF):
                            if self.check_C_exist(bp_stmts[check_id].stmts, touch=True):
                                expand_IF = True
                                # store
                                assert check_IF_id is None
                                check_IF_id = check_id
                        else:
                            break
                
                    # case 3 as well as add if
                    if not expand_IF:
                        # TODO: double check
                        # can have this due to limited time steps
                        if c_stmts is not None:
                            continue
                        # assert c_stmts is None
                        # self.add_if_branch(copy.deepcopy(p), eval_reward, copy.deepcopy(eval_robot), store_cost=store_cost)
                        self.add_if_branch(copy.deepcopy(p), eval_reward, copy_robot(eval_robot), store_cost=store_cost)
                        continue

                # 4) Find C
                c_cond, cond_type = p.find_c_cond(c_touch=True)
                if c_stmts is None:
                    if expand_IF:
                        log_and_print('invalid if branch added and drop for now \n {}'.format(str(p)))
                        continue
                    else:
                        pdb.set_trace()
                        log_and_print('Special case appear and drop for now\n {}'.format(str(p)))
                        continue

                # drop if main branch cannot be entered
                if p.set_unit:
                    if iter >0 and get_structural_cost(p)<=1 and cond_type is None and robot_seed==self.seed:
                        # pdb.set_trace()
                        log_and_print('main branch cannot be entered and drop\n')
                        continue
                    # pass
                else:
                    if iter == 0 and cond_type is None:
                        # pdb.set_trace()
                        log_and_print('main branch cannot be entered and drop\n')
                        break

                # for debug
                if expand_IF:
                    if cond_type != 'i' and cond_type != 'w':
                        # pdb.set_trace()
                        # put it back and try again
                        # due to prob mode
                        # log_and_print('due to prob mode error, put {} back'.format(str(p)))
                        log_and_print('for door key problem: {}'.format(str(p)))
                        # p.reset_c_touch()
                        # p.reset_resume_points()
                        # self.add_queue(p, eval_reward, self.get_robot(robot_seed))
                        continue
                    # assert cond_type == 'i'
                
                # Delete (case 2)
                # consider special
                # TODO: can use smarter way to handle, now just skip it
                if expand_IF and isinstance(tmp_action, ACTION) and tmp_action.abs_state is None:
                    log_and_print('should not get here now')
                    pdb.set_trace()
                    log_and_print('Special case appear and drop for now\n {}'.format(str(p)))
                    continue

                if (expand_IF and cond_type == 'i') and isinstance(tmp_action, ACTION) and len(c_stmts) > 2:
                    # find all satisfied actions
                    cand_idx = None
                    for idx in range(bp_idx, len(bp_stmts)):
                        cand_action = bp_stmts[idx]
                        abs_state_check = (isinstance(cand_action, ACTION) and satisfy_abs_state(tmp_action.post_abs_state, cand_action.post_abs_state)) or \
                            (isinstance(cand_action, HIDE_ACTION) and satisfy_abs_state(tmp_action.post_abs_state, cand_action.action.post_abs_state))
                        action_check = (isinstance(cand_action, ACTION) and str(tmp_action) == str(cand_action)) or \
                                        (isinstance(cand_action, HIDE_ACTION) and str(tmp_action) == str(cand_action.action))
                        if action_check and abs_state_check:
                        # if str(tmp_action) == str(bp_stmts[bp_idx]) and abs_state_check:
                            cand_idx = idx
                            break

                    # end if or add else
                    if cand_idx is not None:
                        pdb.set_trace()
                        # drop
                        # next work on here
                        new_cand_idx = self.case_two_drop(p, cur_action, prev_abs_state, cur_abs_state, bp_idx, bp_stmts, c_idx, c_cond, c_stmts,\
                                                    eval_reward, check_IF_id, new_robot, robot_seed)
                        if new_cand_idx is not None:
                            continue
                        else:
                            pdb.set_trace()
                            print('should not get here')
                            continue

                # Delete (while finish)
                action_num = 0
                for code in c_stmts:
                    if isinstance(code, ACTION):
                        action_num += 1

                if cond_type == 'w' and eval_robot.execute_single_cond(c_cond) and action_num>0:
                    w_stmt = p.find_c_stmt(cond_type='w', c_touch=True)
                    # TODO: check whether use
                    if w_stmt.robot_move:
                        # pdb.set_trace()
                        # print('should not get here')
                        log_and_print('switch seed causes this')

                # Expand (drop when while;C)
                # set resume point
                # p.set_resume_points()

                # expand
                p_list, action_list = p.expand_actions(c_touch=True, keep_touch=True, while_drop= cond_type is None)
                index = list(range(len(p_list)))
                if self.shuffle_actions:
                    np.random.shuffle(index)

                # for either IF or WHILE
                for i in index:
                    # get action and program
                    cur_action = action_list[i]
                    candidate = p_list[i]

                    # check if case two apply
                    if expand_IF:
                        c_stmts, c_idx = candidate.find_actions(c_touch=True)
                        bp_stmts, bp_idx = candidate.find_break_point()

                    if expand_IF and cond_type == 'i' and cur_action is not None and len(c_stmts) > 2:
                        # execute once
                        new_robot = copy_robot(eval_robot)
                        new_robot.execute_single_action(cur_action.action)
                        cur_abs_state = get_abs_state(new_robot)
                        prev_abs_state = get_abs_state(eval_robot)
                        # drop
                        cand_idx = self.case_two_drop(candidate, cur_action, prev_abs_state, cur_abs_state, bp_idx, bp_stmts, c_idx, c_cond, c_stmts,\
                                                    eval_reward, check_IF_id, new_robot, robot_seed)
                        if cand_idx is not None:
                            continue

                    # attempt to drop C for while finish
                    elif (not expand_IF or cond_type=='w') and cur_action is not None and c_cond is not None:
                        # execute once
                        # pdb.set_trace()
                        new_robot = copy_robot(eval_robot)
                        new_robot.execute_single_action(cur_action.action)
                        # end_pos = new_robot.checker.get_hero_pos(new_robot.get_state())
                        end_pos = new_robot.xy.tolist()
                        w_stmt = candidate.find_c_stmt(cond_type='w', c_touch=True)
                        # if str(candidate) == ' move WHILE(not (right_is_clear)) { turn_left move pick_marker C } ; C ; END':
                        #     pdb.set_trace()
                        if w_stmt is not None and end_pos != w_stmt.start_pos and new_robot.execute_single_cond(c_cond):
                            c_stmts, c_idx = candidate.find_actions(c_touch=True)
                            c_stmts.pop(c_idx)

                    candidate.reset_c_touch()
                    # debug
                    c_stmts, c_idx = candidate.find_actions(c_touch=True)
                    assert c_stmts is None
                    # add back
                    # new_robot = copy.deepcopy(eval_robot)
                    new_robot = self.get_robot(robot_seed)
                    new_robot.active = True
                    if store_cost is None:
                        cost, new_reward, add_success = self.add_queue(candidate, eval_reward, new_robot)
                    else:
                        cost, new_reward, add_success = self.add_queue(candidate, eval_reward, new_robot, cost=store_cost+1)
                    # cost, add_success = self.add_queue(candidate, eval_reward, self.get_robot(self.seed))

                    # log print
                    if cur_action is None:
                        log_and_print('attempt to drop:')
                    if add_success:
                        log_and_print('expand: put into queue and get cost {} reward {} for seed {} for\n {}'.format(cost, new_reward, robot_seed, str(candidate)))
                    else:
                        log_and_print('unable to put into queue and get cost {} reward {} for seed {} for\n {}'.format(cost, new_reward, robot_seed, str(p)))
                    continue

                # log print
                log_and_print('------ end current search -------\n')

            if not restart:
                print('[success programs]')
                for reward, program in self.candidates['success']:
                    print(program)
                break
        
        self.pool.close()
        self.pool.terminate()
        self.pool.join()
        # debug store
        # np.save('debug_store/debug.npy', debug_store)

        return self.all_rewards, self.eval_rewards, self.timesteps

