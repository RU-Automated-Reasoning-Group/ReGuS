# algorithm to get result during search
# from search import Node

import numpy as np

# from karel.robot import KarelRobot
from search import Node as Node1
from search2 import Node as Node2
from utils.logging import log_and_print

# from search_enum import EnumNode


class SearchAlg:
    def __init__(self, task='topOff', seed=123, more_seeds=[999, 321], eval_seeds=[0,1], \
                       max_search_iter=100, max_structural_cost=20, shuffle_actions=True, enum=False, logic_expr=None, post_condition=None, allow_action_first=False, multi_seed_eval=False, step_limit_per_sketch=1e9):
        # init
        self.task = task
        self.seed = seed
        self.more_seeds = more_seeds
        self.eval_seeds = eval_seeds
        self.max_search_iter = max_search_iter
        self.max_structural_cost = max_structural_cost
        self.shuffle_action = shuffle_actions
        self.enum = enum
        self.logic_expr = logic_expr
        self.post_condition = post_condition
        self.allow_action_first = allow_action_first
        self.multi_seed_eval = multi_seed_eval
        self.step_limit_per_sketch = step_limit_per_sketch

    def get_result(self, sketch):
        # attempt to fill in sketch
        
        if self.enum:
            node = None #EnumNode(sketch=sketch, task=self.task, seed=self.seed, more_seeds=self.more_seeds, eval_seeds=self.eval_seeds, \
                       # max_search_iter=self.max_search_iter, max_structural_cost=self.max_structural_cost, shuffle_actions=self.shuffle_action, \
                        #found_one=True)
        else:          
            # num_of_while = str(sketch).count('WHILE')
            # search_iters = num_of_while * self.max_search_iter
            search_iters = self.max_search_iter
            if self.multi_seed_eval:
                node = Node2(sketch=sketch, task=self.task, seed=self.seed, more_seeds=self.more_seeds, eval_seeds=self.eval_seeds, \
                        max_search_iter=search_iters, max_structural_cost=self.max_structural_cost, \
                        shuffle_actions=self.shuffle_action, found_one=True, logic_expr=self.logic_expr, post_condition=self.post_condition, allow_action_first=self.allow_action_first)
            else:
                node = Node1(sketch=sketch, task=self.task, seed=self.seed, more_seeds=self.more_seeds, eval_seeds=self.eval_seeds, \
                        max_search_iter=search_iters, max_structural_cost=self.max_structural_cost, \
                        shuffle_actions=self.shuffle_action, found_one=True, logic_expr=self.logic_expr, post_condition=self.post_condition, allow_action_first=self.allow_action_first, step_limit_per_sketch=self.step_limit_per_sketch)        
        # import pdb
        # pdb.set_trace()
        # while_count = str(sketch).count("WHILE")
        # if while_count > 2:
            # return -1, node.candidates
        
        node.search()
        node.clean()
        # get result
        if len(node.candidates['success']) > 0:
            return 1, node.candidates
        elif len(node.candidates['success_search']) > 0:
            log_and_print('fail but success in search')
            for prog in node.candidates['success_search']:
                log_and_print('{}  {}'.format(prog[0], str(prog[1])))
            return 0.8, node.candidates
        # elif len(node.candidates['complete']) > 0:
            # complete_score = np.max([prog[0] for prog in node.candidates['complete']])
            # return 0.5 + 0.3 * complete_score, node.candidates
        # elif len(node.candidates['no_fuel']) > 0:
            # complete_score = np.max([prog[0] for prog in node.candidates['no_fuel']])
            # return 0.3 * complete_score, node.candidates
        else:
            return 0, node.candidates