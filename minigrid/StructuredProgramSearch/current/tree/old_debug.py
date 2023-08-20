# NOTE: node in the tree, contains a sketch and a queue for searching
import copy
import queue
import random
import time
from queue import PriorityQueue

import numpy as np
from dsl import *

from karel.robot import KarelRobot


def get_structural_cost(rules:str):

    # examples
    # rules = '|->S->W'
    # rules = '|->S->W->C'
    # rules = '|->S->W->C->1'

    cost = 0
    for rule in rules.split('->'):
        if rule in ['|', 'S', 'W', 'C']:
            pass
        elif rule in [str(e) for e in range(len(ACTION_LIST))]:
            cost += 1

    return cost


# NOTE: each node is associated with a sketch
class Node:
    def __init__(self, 
            sketch, 
            task, 
            seed=123,
            more_seeds=[],
            max_search_iter = 50,
            max_structural_cost = 30,
            shuffle_actions=False):
        
        self.sketch = sketch
        self.task = task
        self.seed = seed
        self.more_seeds = more_seeds
        self.max_search_iter = max_search_iter
        self.max_structural_cost = max_structural_cost
        self.shuffle_actions = shuffle_actions

        self.robot = KarelRobot(self.task, seed=self.seed)

        self.q = PriorityQueue()
        self.q.put((
            0,  # reward
            0,  # structure cost
            time.time(), # timestamp
            copy.deepcopy(self.sketch),
            {
                'robot': copy.deepcopy(self.robot),
                'rules': '|->S',   # chain of production rules
            }
        ))
 
        self.candidates = {  # store success (and unsuccess) programs
            'success': [],
            'failed': [],
            'no_fuel': [],
            'complete': []
        }

        # TODO: # number of top-k programs we select from a node
        self.k = 1

    def eval_program(self, seed, candidate, tmp_cost, tmp_rules, check_multiple):
        
        success = False
        single_seed = len(self.more_seeds) == 0
        check_multiple = False if single_seed else check_multiple
     
        eval_robot = KarelRobot(task=self.task, seed=seed)
        candidate.execute(eval_robot)
        r = eval_robot.check_reward()

        # force_eval_robot = KarelRobot(task=self.task, seed=seed)
        # force_eval_robot.force_execution = True
        # candidate.execute(force_eval_robot)
        # candidate.reset()  # NOTE: we should deepcopy candidate here, otherwise candidate.complete() may be wrong
        # force_eval_robot.force_execution = False
        # f_r = force_eval_robot.check_reward()
        f_r = 0

        if r == 1 or f_r == 1:
            print('[we found it, w]', candidate)
            if check_multiple:
                passed = True
                for e in self.more_seeds:
                    force_eval_robot = KarelRobot(task=self.task, seed=e)
                    force_eval_robot.force_execution = True
                    candidate.execute(force_eval_robot)
                    candidate.reset()
                    force_eval_robot.force_execution = False
                    passed = passed and force_eval_robot.check_reward() == 1
                    if not passed:
                        print('[failed on other tests]')
                        self.eval_program(e, candidate, tmp_cost, tmp_rules, check_multiple=False)
                        break
                # passed = True
                # for e in self.more_seeds:
                #     passed = passed and self.eval_program(e, candidate, tmp_cost, tmp_rules, check_multiple=False)
                #     if not passed:
                #         break
                if passed:
                    self.candidates['success'].append((1, candidate))
            else:
                # avoid insert duplicate programs
                success = True
                if single_seed:
                    self.candidates['success'].append((1, candidate))
        elif r == -1:
            print('[failed, w]', candidate)
            self.candidates['failed'].append((-1, candidate))
        elif eval_robot.no_fuel():
            print('[no fuel, w]')
            self.candidates['no_fuel'].append((r, candidate))
        elif candidate.complete():
            print('[complete, w]')
            self.candidates['complete'].append((r, candidate))
        else:
            
            # reactivate robot & reset program
            eval_robot.active = True
            candidate.reset()

            # find break point
            bp_stmts, bp_idx = candidate.find_break_point()
            #print('[debug]', candidate)
            bp = bp_stmts[bp_idx]

            # determine cond at break_point IF
            diff_conds = get_diff_conds(bp.abs_state, bp.obs_abs_state)
            
            # insert IF(cond) {C} at break point
            for j in range(len(diff_conds)):
                if j == 0:
                    bp_stmts.insert(bp_idx, IF(cond=diff_conds[j]))
                    print('[get a new program, first, w]', candidate)
                else:
                    bp_stmts[bp_idx] = IF(cond=diff_conds[j])
                    print('[get a new program, more than one, w]', candidate)
                q_tuple = (
                    -r,
                    tmp_cost,
                    time.time(),
                    copy.deepcopy(candidate),
                    {
                        'robot': eval_robot,
                        'rules': tmp_rules,
                    }
                )
                if not tmp_cost > self.max_structural_cost:
                    self.q.put(q_tuple)

        return success

    def search(self):
        
        for iter in range(self.max_search_iter):

            print('[ITER]', iter)

            # 1) get program, find c and its cond
            try:
                r, c, ts, p, game = self.q.get_nowait()
            except queue.Empty:
                break
            else:
                robot, rules = game['robot'], game['rules']
                c_cond, cond_type = p.find_c_cond()
                # NOTE: for sequential C program
                # WHILE(xxx) {action_1} ; C ; END
                # c_cond is None, cond_type is also None
                # TODO: handle this situation
                if c_cond is None:
                    c_stmts, c_index = p.find_seq_c()
                    assert c_stmts is not None
                if c >= self.max_structural_cost:
                    continue

            # 2) expand actions
            p_list = p.expand_actions()
            index = list(range(len(WRAPPED_ACTION_LIST)))
            if self.shuffle_actions:
                random.shuffle(index)

            for p_i in index:
                print(p_list[p_i])
            
            # 3) ranking and put
            for i in index:

                print('[' + str(i) + ']')
                
                _robot = copy.deepcopy(robot)  # one-time-use for evaluation

                tmp_rules = rules + '->' + str(i)
                tmp_cost = get_structural_cost(tmp_rules)
                tmp_action = copy.deepcopy(WRAPPED_ACTION_LIST[i])

                # get abs_state
                tmp_abs_state = get_abs_state(_robot)
                tmp_r = _robot.execute_single_action(tmp_action.action)
                tmp_post_abs_state = get_abs_state(_robot)

                # working on candidate program
                candidate = p_list[i]

                # TODO: we don't handle this for now
                # NOTE: success on-the-fly or failed on-the-fly
                # if tmp_r == 1:
                    
                #     # find C
                #     c_stmts, c_idx = candidate.find_actions()
                #     tmp_action.abs_state = tmp_abs_state
                #     tmp_action.post_abs_state = tmp_post_abs_state
                #     c_stmts[c_idx-1] = tmp_action
                #     c_stmts.pop(c_idx)  # remove C

                #     bp_stmts, bp_idx = candidate.find_break_point()
                #     if not bp_stmts is None:
                #         bp = bp_stmts[bp_idx]

                #         # special case: may be case one
                #         if cond_type == 'i' and len(c_stmts) == 1 and tmp_action.action.action == bp.action.action:
                #             bp_stmts.pop(bp_idx-1)
                #             current_abs_state = get_abs_state(_robot)
                #             bp.abs_state = merge_abs_state(bp.abs_state, tmp_abs_state)
                #             bp.obs_abs_state = tmp_abs_state
                    
                #     self.eval_program(self.seed, copy.deepcopy(candidate), tmp_cost, tmp_rules, check_multiple=True)
                #     continue

                # elif tmp_r == -1:
                #     self.candidates['failed'].append(candidate)
                #     continue

                simple_expand = None

                if cond_type == 'w':

                    if not _robot.execute_single_cond(c_cond):
                        print('[should expand]')
                        simple_expand = True

                    else:
                        c_stmts, c_idx = candidate.find_actions()
                        tmp_action.abs_state = tmp_abs_state
                        tmp_action.post_abs_state = tmp_post_abs_state
                        c_stmts[c_idx-1] = tmp_action
                        c_stmts.pop(c_idx)  # remove C
                        tmp_c_cond, tmp_cond_type = candidate.find_c_cond()
                        if tmp_c_cond is None:
                            self.eval_program(self.seed, copy.deepcopy(candidate), tmp_cost, tmp_rules, check_multiple=True)
                        else:
                            # should restart newly completed WHILE
                            print('[should restart]')
                            update_robot = KarelRobot(task=self.task, seed=self.seed)
                            update_robot.force_execution = True
                            candidate.execute(update_robot)
                            update_robot.force_execution = False
                            candidate.reset()
                            _robot = update_robot
                            simple_expand = True

                elif cond_type == 'i':

                    # find break point
                    bp_stmts, bp_idx = candidate.find_break_point()
                    bp = bp_stmts[bp_idx]

                    # find C
                    c_stmts, c_idx = candidate.find_actions()
                    
                    # handle case one here
                    if len(c_stmts) == 2 and tmp_action.action.action == bp.action.action:
                        bp_stmts.pop(bp_idx-1)
                        current_abs_state = get_abs_state(_robot)
                        bp.abs_state = merge_abs_state(bp.abs_state, tmp_abs_state)
                        bp.obs_abs_state = tmp_abs_state
                        bp.break_point = False  # TODO: not so sure

                        self.eval_program(self.seed, copy.deepcopy(candidate), tmp_cost, tmp_rules, check_multiple=True)

                    else:
                        tmp_action.abs_state = tmp_abs_state
                        tmp_action.post_abs_state = tmp_post_abs_state
                        c_stmts[c_idx-1] = tmp_action

                        # try to connect to the break point
                        future_robot = copy.deepcopy(_robot)  # for one time use
                        current_abs_state = get_abs_state(future_robot)
                        future_robot.execute_single_action(bp.action)  # execute break_point action
                        future_abs_state = get_abs_state(future_robot)
                            
                        if not satisfy_abs_state(future_abs_state, bp.post_abs_state):
                            simple_expand = True

                        else:
                            print('terminate an IF')
                            bp.abs_state = merge_abs_state(bp.abs_state, current_abs_state)
                            bp.break_point = False
                            c_stmts.pop(c_idx)  # remove C

                            tmp_c_cond, tmp_cond_type = candidate.find_c_cond()
                            if tmp_c_cond is None:
                                self.eval_program(self.seed, copy.deepcopy(candidate), tmp_cost, tmp_rules, check_multiple=True)
                            else:
                                # should restart newly completed WHILE
                                update_robot = KarelRobot(task=self.task, seed=self.seed)
                                update_robot.force_execution = True
                                candidate.execute(update_robot)
                                update_robot.force_execution = False
                                candidate.reset()
                                _robot = update_robot
                                simple_expand = True

                # TODO: handle sequential C situation
                elif cond_type is None:
                    simple_expand = True

                if simple_expand:
                    c_stmts, c_idx = candidate.find_actions()
                    c_stmts[c_idx-1].abs_state = tmp_abs_state
                    c_stmts[c_idx-1].post_abs_state = tmp_post_abs_state
                    
                    q_tuple = (
                        -tmp_r,  # primary: retreive item with the highest reward
                        tmp_cost,  # secondary: retreive item with the lowest cost
                        time.time(),  # the final evaluation metric
                        copy.deepcopy(candidate),
                        {
                            'robot': _robot,
                            'rules': tmp_rules,  # TODO: should we include IF in rules?
                        }
                    )
                    
                    if not tmp_cost > self.max_structural_cost:
                        self.q.put(q_tuple)

        # TODO: tmp code for printing
        #print(self.candidates)
        #print(self.q.qsize())
        print('[success programs]')
        for reward, program in self.candidates['success']:
            print(program)


if __name__ == "__main__":

    random.seed(123)
    np.random.seed(123)

    # NOTE: for simplicity, not a tree right now
    program_db = []

    p = Program()
    program_db.append(p)

    _p = copy.deepcopy(program_db[0])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[1])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[2])
    program_db += _p.expand()

    _p = copy.deepcopy(program_db[3])
    program_db += _p.expand()

    
    # task    : topOff
    # program : WHILE(front_is_clear) { C } ; S ; END
    example_program = program_db[-2]
    #more_seeds = [999, 123, 666, 546, 11, 4372185, 6431, 888, 1, 2, 3, 4, 5]
    #more_seeds = [999, 123, 666, 11, 4372185, 6431, 888, 1, 2, 3, 4, 5]
    #more_seeds = [999, 321, 1234567]
    #more_seeds = [546]
    more_seeds = []
    node = Node(sketch=example_program, task='topOff', seed=12, more_seeds=more_seeds, max_search_iter=1000, max_structural_cost=20, shuffle_actions=True)
    node.search()
    exit()

    # task    : stairClimber
    # program : WHILE(not (front_is_clear)) { C } ; S ; END
    # example_program = program_db[7].expand()[0]
    # more_seeds = [1, 123, 432, 84314, 73]
    # node = Node(sketch=example_program, task='stairClimber', seed=321, more_seeds=more_seeds, max_search_iter=100, max_structural_cost=20, shuffle_actions=True)
    # node.search()
    # exit()
    
    # task    : fourCorner
    # program : WHILE(not (markers_present)) { WHILE(front_is_clear) { C } ; C } ; S ; END
    # example_program = copy.deepcopy(program_db[-3]).expand()[-1].expand()[0].expand()[0].expand()[0]
    # node = Node(sketch=example_program, task='fourCorners', seed=123, more_seeds=[], max_search_iter=100, max_structural_cost=20, shuffle_actions=True)
    # node.search()
    # exit()

    # task    : randomMaze
    # program : WHILE(not (markers_present)) { WHILE(front_is_clear) { C } ; C } ; S ; END
    # example_program = copy.deepcopy(program_db[-3]).expand()[-1].expand()[0].expand()[0].expand()[0]
    # more_seeds = [999, 123, 666, 546, 11]
    # node = Node(sketch=example_program, task='randomMaze', seed=321, more_seeds=more_seeds, max_search_iter=1000, max_structural_cost=40, shuffle_actions=False)
    # node.search()
    # exit()

    # TODO: simpler program
    # task    : randomMaze
    # seed    : 0
    ########
    ########
    #...#.##
    ###1#.##
    #.#...##
    #.#.####
    #>....##
    ########
    # program : WHILE(not (markers_present)) { C } ; S ; END
    example_program = copy.deepcopy(program_db[-3]).expand()[0]
    more_seeds = [999, 123, 666, 546, 11, 4372185, 6431, 888, 1, 2, 3, 4, 5, 0]
    more_seeds = [0]
    more_seeds = []
    seed = 0
    node = Node(sketch=example_program, task='randomMaze', seed=seed, more_seeds=more_seeds, max_search_iter=100, max_structural_cost=40, shuffle_actions=True)
    node.robot.draw()
    node.search()
    exit()

    # TODO: try to see if sequential C work
    # task    : topOff
    # program : WHILE(front_is_clear) { C } ; C ; END
    example_program = program_db[-2].expand()[0]
    node = Node(sketch=example_program, task='topOff', seed=123, more_seeds=[999, 321, 1234567], max_search_iter=100, max_structural_cost=20, shuffle_actions=True)
    node.search()
    exit()