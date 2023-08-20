# NOTE: node in the tree, contains a sketch and a queue for searching
import time
from copy import deepcopy
from queue import PriorityQueue

import numpy as np
from dsl import *

from karel.robot import KarelRobot


# NOTE: each node is associated with a sketch
class Node:
    def __init__(self, sketch, task):
        
        self.sketch = sketch

        self.task = task
        self.robot = KarelRobot(self.task, seed=999)  # seed is not important at this time

        self.q = PriorityQueue()
        self.q.put((
            0,  # reward 
            0,  # structure cost
            time.time(), # timestamp
            {
                'program': copy.deepcopy(self.sketch), 
                'robot': copy.deepcopy(self.robot),
                'rules': '|->S',   # chain of production rules
            }
        ))

        self.candidates = []
        self.k = 1  # number of programs we select from a node

    def search(self):
        pass


def get_structural_cost(rules:str):

    # examples
    # rules = '|->S->W'
    # rules = '|->S->W->C'
    # rules = '|->S->W->C->1'

    cost = 0
    for rule in rules.split('->'):
        if rule in ['|', 'S', 'W', 'C']:
            pass
        elif rule in [str(e+1) for e in range(len(ACTION_LIST))]:
            cost += 1

    return cost


if __name__ == "__main__":
    
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

    # NOTE: WHILE(front_is_clear) { C } ; S
    example_program = program_db[-2]
    
    node = Node(sketch=example_program, task='topOff')

    for iter in range(6):

        # 1) get item
        r, c, ts, game = node.q.get()
        p, robot, rules = game['program'], game['robot'], game['rules']
        print(rules)
        
        # 1.5) find actions and WHILE?
        # TODO: this can be empty, handle this later
        # NOTE: the important part is w.cond[0]
        #_, _, w = p.find_actions_while()
        
        # TODO: handle this
        # WHILE(front_is_clear) { IF(not (no_markers_present)) { move C }  move} ; S
        # should find cond instead, can be WHILE or IF
        c_cond, cond_type = p.find_c_cond()

        # 2) expand actions
        # TODO: check if p_list is empty 
        p_list = p.expand_actions()
        if not p_list:
            break
        
        # 3) ranking and put
        for i in range(len(ACTION_LIST)):
            tmp_robot = copy.deepcopy(robot)
            tmp_robot_copy = copy.deepcopy(robot)
            tmp_rules = rules + '->' + str(i)
            tmp_cost = get_structural_cost(game['rules'])
            tmp_action = copy.deepcopy(WRAPPED_ACTION_LIST[i])
            
            tmp_action.abs_state = get_abs_state(tmp_robot)
            tmp_r = tmp_robot.execute_single_action(tmp_action.action)
            tmp_action.post_abs_state = get_abs_state(tmp_robot)
            #tmp_robot.draw()
            
            print('[debug]', i, p_list[i])

            if tmp_robot.execute_single_cond(c_cond) and cond_type == 'w':
                
                print('[terminate current branch]')

                c_stmts, c_idx = p_list[i].find_actions()
                c_stmts[c_idx-1] = tmp_action
                                               # NOTE: this is dangerous...
                c_stmts.pop(c_idx)
                candidate = p_list[i]

                #tmp_robot.draw()
                # TODO: duplicate actions?
                candidate.execute(tmp_robot)
                #candidate.execute(tmp_robot_coy)
                r = tmp_robot.check_reward()
                if r == 1:
                    print('[we found it]')
                    node.candidates.append((1, candidate))
                elif tmp_robot.no_fuel():
                    print('[no fuel]')
                    node.candidates.append((r, candidate))
                    break
                #tmp_robot.draw()

                # NOTE: we should reactivate robot
                tmp_robot.active = True

                # find & deactivate break point
                stmts, idx = candidate.find_break_point()
                stmts[idx].break_point = False

                # determine cond in break_point IF
                bp = stmts[idx]
                conds = diff_abs_state(bp.abs_state, bp.diff_abs_state)
                
                # insert IF(cond) {C} at break point
                for j in range(len(conds)):
                    if j == 0:
                        stmts.insert(idx, IF(cond=conds[j]))
                        print('[get a new program]', candidate)
                    else:
                        stmts[idx] = IF(cond=conds[j])
                        print('[get a new program]', candidate)
                    
                    q_tuple = (
                        #-tmp_r,  # primary: retreive item with the highest reward
                        # TODO: check this later
                        r,
                        tmp_cost,  # secondary: retreive item with the lowest cost
                        time.time(),
                        {
                            'program': candidate,
                            'robot': tmp_robot,
                            'rules': tmp_rules
                        }
                    )
                    node.q.put(q_tuple)
            elif tmp_robot.execute_single_cond(c_cond) and cond_type == 'i':

                # TODO: after put marker, I should.... change move in branch to DC
                #     : more specifically, move has a change to change abs_state from T/F to DC
                c_stmts, c_idx = p_list[i].find_actions()
                c_stmts[c_idx-1] = tmp_action  # TODO: includes abs_state
                                               # NOTE: this is dangerous...
                c_stmts.pop(c_idx)
                candidate = p_list[i]
                print('?', tmp_robot.check_reward())
                if tmp_r == 0.5:
                    print('okay')
                    candidate.stmts[0].stmts[1].abs_state.state = {
                        'front_is_clear'    : 'T',
                        'left_is_clear'     : 'T',
                        'right_is_clear'    : 'F',
                        'no_markers_present': 'DC',
                    }
                candidate.execute(tmp_robot_copy)
                tmp_robot_copy.draw()
                r = tmp_robot_copy.check_reward()
                print('sub branch?', candidate, r)

                # TODO: debug this
                if tmp_r == 0.5:
                    print('[tmp_r==0.5][debug]')
                    print(candidate.stmts[0].stmts[1].abs_state.state)
                    #exit()


                #tmp_robot.draw()
                if r == 1:
                    print('ok?')
                    exit()
                                
            else:
                c_stmts, c_idx = p_list[i].find_actions()
                c_stmts[c_idx-1] = tmp_action  # TODO: includes abs_state
                                               # NOTE: this is dangerous...
                
                q_tuple = (
                    -tmp_r,  # primary: retreive item with the highest reward
                    tmp_cost,  # secondary: retreive item with the lowest cost
                    time.time(),
                    {
                        'program': p_list[i],
                        'robot': tmp_robot,
                        'rules': tmp_rules
                    }
                )

                node.q.put(q_tuple)

    print('[size]', node.q.qsize())
    print('[candidates]', node.candidates)

    # TODO: abs_state
    # wrapper for k_action?

    # TODO: reward for each enumeration?
    # TODO: next step is to include abs_state
                