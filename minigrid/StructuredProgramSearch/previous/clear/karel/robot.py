import copy
import torch
import numpy as np

from .karel import Karel
from .generator import KarelStateGenerator
from .checker import *
from .dsl import *


# single karel robot
class KarelRobot:
    def __init__(self, task, seed=999):
        
        self.task = task
        self.seed = seed
        self.gen = KarelStateGenerator(seed=self.seed)

        # init state
        self.init_state = self.state_init(self.gen, self.task)
        self.karel = Karel(state=self.init_state)

        # init checker
        self.checker = self.checker_init(self.task)

        # TODO: max_steps is pre-set now
        self.steps = 0
        self.max_steps = 50
        self.no_fuel = False

        # under construction
        # only for accumulating rewards when restarting while loop
        self.acc_reward = 0
        self.is_accumulated = False

    def start_acc(self):
        self.is_accumulated = True

    def acc(self, reward):
        self.acc_reward += reward

    def end_acc(self):
        acc_reward = self.acc_reward
        
        self.acc_reward = 0
        self.is_accumulated = False
        
        return acc_reward

    def state_init(self, gen, task):

        gen_function_base = "generate_single_state_"

        if task == "cleanHouse":
            gen_function = gen_function_base + "clean_house"
        elif task == "harvester":
            gen_function = gen_function_base + "harvester"
        elif task == "randomMaze":
            gen_function = gen_function_base + "random_maze"
        elif task == "fourCorners":
            gen_function = gen_function_base + "four_corners"
        elif task == "stairClimber":
            gen_function = gen_function_base + "stair_climber"
        elif task == "topOff":
            gen_function = gen_function_base + "top_off"
        else:
            print('Please check the task')
            exit()

        state, _, _, _, _ = getattr(gen, gen_function)()
        
        return state.astype(int)

    def checker_init(self, task):

        if task == "cleanHouse":
            checker = CleanHouseChecker(self.init_state)
        elif task == "harvester":
            checker = HarvesterChecker(self.init_state)
        elif task == "randomMaze":
            checker = RandomMazeChecker(self.init_state)
        elif task == "fourCorners":
            checker = FourCornerChecker(self.init_state)
        elif task == "stairClimber":
            checker = StairClimberChecker(self.init_state)
        elif task == "topOff":
            checker = TopOffChecker(self.init_state)

        return checker

    def get_state(self):
        return self.karel.state

    def execute_single_action(self, action):
        assert isinstance(action, (k_action, k_end))
        
        # return reward
        return action(self.karel, self)

    def execute_single_cond(self, cond):
        assert isinstance(cond, k_cond)

        return cond(self.karel)

    def check_reward(self):
        return self.checker(self.get_state())

    def draw(self, *args, **kwargs):
        return self.karel.draw(*args, **kwargs)



class BatchedKarelRobots:
    def __init__(self, task, batch_size, seed=999):

        self.task = task
        self.batch_size = batch_size
        self.seed = seed

        self.robots = []
        for i in range(self.batch_size):
            self.robots.append(KarelRobot(task=self.task, seed=self.seed + i))

    def execute_single_action(self, action, candidates=None):
        candidates = list(range(self.batch_size)) if candidates is None else candidates
        total_reward = 0.0
        for i in candidates:
            r = self.robots[i].execute_single_action(action)
            total_reward += r

        return total_reward / len(candidates)

    def execute_single_cond(self, cond, candidates=None):
        candidates = list(range(self.batch_size)) if candidates is None else candidates
        true_candidates, false_candidates = [], []
        for index in candidates:
            if self.robots[index].execute_single_cond(cond):
                true_candidates.append(index)
            else:
                false_candidates.append(index)

        return true_candidates, false_candidates

    def get_latent_state(self, encoder, candidates=None):
        candidates = list(range(self.batch_size)) if candidates is None else candidates
        ph_states_tensors = []
        for i in candidates:
            ph_states_tensors.append(torch.from_numpy(self.robots[i].get_state()))
        stacked_states = torch.stack(ph_states_tensors, dim=0)
        
        return encoder(stacked_states.float())
    
    def any_no_fuel(self):
        no_fuel = False
        for robot in self.robots:
            no_fuel = no_fuel or robot.no_fuel
        
        return no_fuel