# algorithm to get result during search
# from search import Node

import numpy as np

from search import Node
from utils.logging import log_and_print

# from search_enum import EnumNode


class SearchAlg:
    def __init__(self, task='topOff', seed=123, more_seeds=[999, 321], eval_seeds=[0,1], \
                       max_search_iter=100, max_structural_cost=20, shuffle_actions=True, enum=False, logic_expr=None):
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
            node = Node(sketch=sketch, task=self.task, seed=self.seed, more_seeds=self.more_seeds, eval_seeds=self.eval_seeds, \
                        max_search_iter=search_iters, max_structural_cost=self.max_structural_cost, \
                        shuffle_actions=self.shuffle_action, found_one=True, logic_expr=self.logic_expr)
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
        elif len(node.candidates['complete']) > 0:
            complete_score = np.max([prog[0] for prog in node.candidates['complete']])
            return 0.5 + 0.3 * complete_score, node.candidates
        elif len(node.candidates['no_fuel']) > 0:
            complete_score = np.max([prog[0] for prog in node.candidates['no_fuel']])
            return 0.3 * complete_score, node.candidates
        else:
            return 0, node.candidates