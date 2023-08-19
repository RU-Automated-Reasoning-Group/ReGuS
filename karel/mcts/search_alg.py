import numpy as np

from dsl_karel import *
from utils.logging import log_and_print
from search_karel import Node

class SearchAlg:
    def __init__(self, task='topOff', seed=123, more_seeds=[999, 321], \
                       max_search_iter=100, max_structural_cost=20, \
                       structural_wegiht=0.2, shuffle_actions=True, goals=[1,]):
        # init
        self.task = task
        self.seed = seed
        self.more_seeds = more_seeds
        self.max_search_iter = max_search_iter
        self.max_structural_cost = max_structural_cost
        self.structural_wegiht = structural_wegiht
        self.shuffle_action = shuffle_actions
        self.goals = goals

    def get_result(self, sketch):
        # attempt to fill in sketch       
        node = Node(sketch=sketch, task=self.task, seed=self.seed, more_seeds=self.more_seeds, \
                    max_search_iter=self.max_search_iter, max_structural_cost=self.max_structural_cost, \
                    structural_weight=self.structural_wegiht, \
                    shuffle_actions=self.shuffle_action, found_one=True, sub_goals=self.goals)
        node.search()

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