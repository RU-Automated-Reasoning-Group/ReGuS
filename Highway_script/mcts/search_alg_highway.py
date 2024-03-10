# algorithm to get result during search
# from search import Node

import numpy as np

from dsl_highway_all import *
# from dsl_highway import *
from utils.logging import log_and_print

from search_highway_new import Node


class SearchAlg:
    def __init__(self, task='highway-fast-v0', seed=123, more_seeds=[999, 321], eval_seeds=[0,1], \
                       max_search_iter=100, max_structural_cost=20, shuffle_actions=True, enum=False, prob_mode=False, \
                       sub_goals=[10,20,30], config=None, more_config=None, eval_config=None):
        # init
        self.task = task
        self.seed = seed
        self.more_seeds = more_seeds
        self.eval_seeds = eval_seeds
        self.max_search_iter = max_search_iter
        self.max_structural_cost = max_structural_cost
        self.shuffle_action = shuffle_actions
        self.enum = enum
        self.prob_mode = prob_mode
        self.sub_goals = sub_goals
        self.config = config
        self.more_config = more_config
        self.eval_config = eval_config

        self.total_iter = []
        self.total_reward = []
        self.total_eval_reward = []

    def get_result(self, sketch, search_iter=None):
        # attempt to fill in sketch
        # get max search iteration
        if search_iter is None:
            search_iter = self.max_search_iter

        # limit last C if highway
        if isinstance(sketch.stmts[-2], C):
            sketch.stmts.pop(-2)

        node = Node(sketch=sketch, task=self.task,
                    seed=self.seed, more_seeds=self.more_seeds, eval_seeds=self.eval_seeds,
                    max_search_iter=search_iter, max_structural_cost=self.max_structural_cost, 
                    shuffle_actions=self.shuffle_action, found_one=True, prob_mode=self.prob_mode,
                    config_set=self.config, more_config_set=self.more_config, eval_config_set=self.eval_config, timesteps_goals=self.sub_goals)


        reward, eval_reward, timesteps = node.search()
        if len(reward) > 0:
            assert len(timesteps) - len(reward) <= 1
            if len(timesteps) > len(reward):
                timesteps = timesteps[:-1]

            if len(self.total_iter) > 0:
                self.total_iter += (np.array(timesteps) + self.total_iter[-1]).tolist()
            else:
                self.total_iter += timesteps
            self.total_reward += reward
            self.total_eval_reward += eval_reward
            
        # get result
        log_and_print('best reward{}\n total time steps{}'.format(max(reward), self.total_iter[-1]))
        return max(reward), node.candidates
