# NOTE: each node is associated with a sketch
import enum
import queue
import time
import copy
import random
from queue import PriorityQueue

import numpy as np

from dsl import *
from karel.robot import KarelRobot

from utils.logging import log_and_print

import pdb

def get_structural_cost(program):
    cost = 0
    program_str = str(program)
    for s in ACTION_DICT:
        cost += program_str.count(s)

    return cost


# NOTE: each node is associated with a sketch
class EnumNode:
    # constant
    SUCCESS_TYPE = 'success'
    FAIL_TYPE = 'fail'
    MORE_WORK_TYPE = 'more_work'

    # init
    def __init__(self, 
            sketch, 
            task, 
            seed=123,
            more_seeds=[],
            eval_seeds=[],
            max_search_iter = 50,
            max_structural_cost = 30,
            shuffle_actions=False,
            found_one=False):
        
        self.sketch = sketch
        self.task = task
        self.seed = seed
        self.more_seeds = more_seeds
        self.eval_seeds = eval_seeds
        self.max_search_iter = max_search_iter
        self.max_structural_cost = max_structural_cost
        self.shuffle_actions = shuffle_actions
        self.found_one = found_one

        # store all required robot
        self.robot_store = {self.seed: KarelRobot(self.task, seed=self.seed)}
        for e in more_seeds:
            self.robot_store[e] = KarelRobot(self.task, seed=e)
        for e in eval_seeds:
            self.robot_store[e] = KarelRobot(self.task, seed=e)
        assert len(eval_seeds) > 0

        self.q = PriorityQueue()
        self.q.put((
            0,  # reward
            time.time(), # timestamp
            copy.deepcopy(self.sketch), # sketch
            0   # cost
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


    # get robot from store
    def get_robot(self, seed):
        return copy.deepcopy(self.robot_store[seed])

    # add candidate program into queue
    def add_queue(self, candidate, reward, cost):
        # add back
        # cost = get_structural_cost(candidate)
        q_tuple = (
            # -reward + cost * 0.02,
            # -reward + cost * 0.04,
            -reward + cost * 0.1,
            time.time(),
            candidate,
            cost
        )
        if not cost > self.max_structural_cost:
            self.q.put(q_tuple)

            return cost, True
        return cost, False

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


    def get_rewards(self, prog, seeds):
        reward_list = []
        for e in seeds:
            # force evaluate
            force_eval_robot = self.get_robot(e)
            force_eval_robot.force_execution = True
            prog.execute(force_eval_robot)
            prog.reset()
            prog.reset_c_touch()
            force_eval_robot.force_execution = False
            reward_list.append(force_eval_robot.check_reward())

        return np.mean(reward_list)


    def eval_program(self, seed, candidate, check_multiple):
        success = False
        single_seed = len(self.more_seeds) == 0
        check_multiple = False if single_seed else check_multiple
     
        # execute and get reward
        eval_robot = self.get_robot(seed)
        eval_robot.force_execution = True
        candidate.execute(eval_robot)
        r = eval_robot.check_reward()

        # if str(candidate) == " WHILE(not (right_is_clear)) { turn_left move turn_right C } ; C ; END":
        #     pdb.set_trace()

        # success
        if r == 1:
            # multiple seed check
            if check_multiple:
                passed = True
                for e in self.more_seeds:
                    # force evaluate
                    force_eval_robot = self.get_robot(e)
                    force_eval_robot.force_execution = True
                    candidate.execute(force_eval_robot)
                    candidate.reset()
                    force_eval_robot.force_execution = False
                    passed = passed and force_eval_robot.check_reward() == 1
                    # attempt to add 
                    if not passed:
                        # log and print
                        log_and_print('\nfound but not success in all seeds for \n {}'.format(candidate))
                        self.candidates['success_search'].append((1, candidate))
                        # TODO: double check
                        eval_result, eval_robot = self.eval_program(e, candidate, check_multiple=False)
                        assert eval_result != self.SUCCESS_TYPE
                        return eval_result, eval_robot
            else:
                # avoid insert duplicate programs
                success = True
                if single_seed:
                    pdb.set_trace()
                    log_and_print('\n success when not check multi for \n {}'.format(candidate))
                    self.candidates['success'].append((1, candidate))

            return self.SUCCESS_TYPE, eval_robot
        # fail
        elif r == -1:
            # log and print
            log_and_print('\n fail for \n {}'.format(candidate))
            self.candidates['failed'].append((-1, candidate))
            return self.FAIL_TYPE, eval_robot
        # no fuel
        elif eval_robot.no_fuel():
            # log and print
            log_and_print('\n no fuel with reward {} for \n {}'.format(r, candidate))
            self.candidates['no_fuel'].append((r, candidate))
            return self.FAIL_TYPE, eval_robot
        # complete
        elif candidate.complete():
            # log and print
            log_and_print('\n complete with reward {} for\n {}'.format(r, candidate))
            self.candidates['complete'].append((r, candidate))
            return self.FAIL_TYPE, eval_robot
        # need additional operation
        else:
            return self.MORE_WORK_TYPE, eval_robot


    # Note:
    # Execute Program:
    # -> Success -> multi. seed -> END
    # -> Fail -> END
    # -> C -> ACTION | None | IF
    def search(self):
        cur_id = 0
        while True:
            restart = False
            # do search
            for iter in range(self.max_search_iter):
                log_and_print('[ITER] {}'.format(iter))

                # get one program
                try:
                    r, ts, p, c = self.q.get_nowait()
                except queue.Empty:
                    break

                # double check: debug
                tmp_c_stmts, _ = p.find_actions(c_touch=True)
                assert tmp_c_stmts is None

                # log print
                log_and_print('searching base on {} with cost {}'.format(str(p), r))

                # Execute Program
                eval_result, eval_robot = self.eval_program(self.seed, p, check_multiple=False)
                eval_reward = eval_robot.check_reward()
                p.reset()

                # get action before C
                c_stmts, c_idx = p.find_actions(c_touch=True)
                tmp_action = None
                if c_stmts is not None and len(c_stmts)>1:
                    tmp_action = c_stmts[c_idx-1]

                # 1) Success
                if eval_result == self.SUCCESS_TYPE:
                    c_stmts, c_idx = p.find_actions()
                    # TODO: doublt check
                    if c_stmts is not None and len(c_stmts)>1:
                        tmp_action = c_stmts[c_idx-1]

                    c_cond, cond_type = p.find_c_cond()
                    set_fail = False

                    # has C
                    if c_stmts is not None:
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
                        eval_result, eval_robot = self.eval_program(self.seed, p, check_multiple=True)
                        if eval_result == self.MORE_WORK_TYPE:
                            eval_reward = eval_robot.check_reward()
                            log_and_print('more work from other seeds')
                            p.reset()
                        elif eval_result == self.SUCCESS_TYPE:
                            test_result = self.test_program(p)
                            if self.found_one and test_result == self.SUCCESS_TYPE:
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
                            if cur_id + 1 >= len(self.more_seeds):
                                break
                            self.seed = self.more_seeds[cur_id + 1]
                            self.q.put((
                                0,  # reward
                                time.time(), # timestamp
                                copy.deepcopy(self.sketch),
                                0
                            ))
                            log_and_print('\nswitch from seed {} to {}'.format(self.more_seeds[cur_id], self.more_seeds[cur_id+1]))
                            cur_id += 1
                            break
                        else:
                            log_and_print('\nfound but not success in all seeds (special) for \n {}'.format(p))
                            continue

                # 2) Fail
                elif eval_result == self.FAIL_TYPE:
                    continue

                # 3) Find C
                if c_stmts is None:
                    pdb.set_trace()
                assert c_stmts is not None

                # drop if main branch cannot be entered
                c_cond, cond_type = p.find_c_cond(c_touch=True)
                if iter == 0 and cond_type is None:
                    # pdb.set_trace()
                    log_and_print('main branch cannot be entered and drop\n')
                    break

                # Expand (drop when while;C)
                p_list, action_list = p.expand_actions(c_touch=True, while_drop=False)
                index = list(range(len(p_list)))
                if self.shuffle_actions:
                    random.shuffle(index)

                # Expand action for either IF or WHILE
                for i in index:
                    # get action and program
                    cur_action = action_list[i]
                    candidate = p_list[i]

                    # evaluate to get new reward
                    cand_reward = self.get_rewards(candidate, self.more_seeds)

                    candidate.reset_c_touch()
                    # debug
                    cand_c_stmts, cand_c_idx = candidate.find_actions(c_touch=True)
                    assert cand_c_stmts is None

                    # add back
                    # cost, add_success = self.add_queue(candidate, eval_reward, c+1)
                    cost, add_success = self.add_queue(candidate, cand_reward, c+1)

                    # log print
                    assert cur_action is not None
                    if add_success:
                        log_and_print('expand: put into queue and get cost {} reward {} for\n {}'.format(cost, eval_reward, str(candidate)))
                    else:
                        log_and_print('unable to put into queue and get cost {} reward {} for\n {}'.format(cost, eval_reward, str(p)))

                # add None
                if len(c_stmts) > 1:
                    cand_p = copy.deepcopy(p)
                    cand_c_stmts, cand_c_idx = cand_p.find_actions(c_touch=True)
                    # drop
                    cand_c_stmts.pop(cand_c_idx)

                    # evaluate to get new reward
                    cand_reward = self.get_rewards(cand_p, self.more_seeds)

                    cand_p.reset_c_touch()
                    # debug
                    cand_c_stmts, cand_c_idx = cand_p.find_actions(c_touch=True)
                    assert cand_c_stmts is None

                    # add back
                    cost, add_success = self.add_queue(cand_p, cand_reward, c)

                    # log print
                    if add_success:
                        log_and_print('expand None: put into queue and get cost {} reward {} for\n {}'.format(cost, eval_reward, str(cand_p)))
                    else:
                        log_and_print('unable to put into queue and get cost {} reward {} for\n {}'.format(cost, eval_reward, str(p)))

                # Expand If
                for cand_cond in COND_LIST:
                    # add if
                    new_if = IF(cond=copy.deepcopy(cand_cond))
                    cand_p = copy.deepcopy(p)
                    cand_c_stmts, cand_c_idx = cand_p.find_actions(c_touch=True)
                    cand_c_stmts.insert(cand_c_idx, new_if)

                    # evaluate to get new reward
                    # cand_reward = self.get_rewards(cand_p, self.more_seeds)

                    cand_p.reset_c_touch()
                    # debug
                    cand_c_stmts, cand_c_idx = cand_p.find_actions(c_touch=True)
                    assert cand_c_stmts is None
                    
                    # add back
                    cost, add_success = self.add_queue(cand_p, eval_reward, c+1)
                    # cost, add_success = self.add_queue(cand_p, cand_reward, c+1)

                    # log print
                    if add_success:
                        log_and_print('expand IF: put into queue and get cost {} reward {} for\n {}'.format(cost, eval_reward, str(cand_p)))
                    else:
                        log_and_print('unable to put into queue and get cost {} reward {} for\n {}'.format(cost, eval_reward, str(p)))

                # log print
                log_and_print('------ end current search -------\n')

            if not restart:
                print('[success programs]')
                for reward, program in self.candidates['success']:
                    print(program)
                break
            elif cur_id + 1 >= len(self.more_seeds):
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

