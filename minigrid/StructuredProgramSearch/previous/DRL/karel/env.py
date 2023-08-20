import copy
import torch
import numpy as np

from .karel import Karel
from .generator import KarelStateGenerator
from .checker import *
from .dsl import *
from network.network import *


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

        # execution trace
        self.history = [self.init_state]

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
        assert isinstance(action, k_action)

        # return reward
        return action(self.karel, self)

    def check_reward(self):
        return self.checker(self.get_state())

    def draw(self, *args, **kwargs):
        return self.karel.draw(*args, **kwargs)


class KarelRobotExecutor:
    def __init__(self, task, batch_size, seed=999):
        
        self.task = task
        self.batch_size = batch_size
        self.seed = seed

        self.robots = []
        for i in range(self.batch_size):
            self.robots.append(KarelRobot(task=self.task, seed=self.seed + i))

        # init the program
        self.program = k_prog()

    def execute_single_action(self, action, index=None):
        exe_range = range(self.batch_size) if index is None else index
        total_reward = 0
        for i in exe_range:
            total_reward += self.robots[i].execute_single_action(action)

        return total_reward / len(list(exe_range))

    def get_latent_state(self, encoder):
        ph_states_tensors = []
        for i in range(self.batch_size):
            ph_states_tensors.append(torch.from_numpy(self.robots[i].get_state()))
        stacked_states = torch.stack(ph_states_tensors, dim=0)
        
        latent_state = encoder(stacked_states.float())
        return latent_state

    def predict(self, predictor, latent_state):

        self.nll = 0
        reward = 0
        done = False

        prediction_index, nll = predictor(latent_state)
        self.nll += nll

        if prediction_index in ACTION_INDEX:
            action = k_action(
                ACTION_NAME[prediction_index]
            )
            reward = self.execute_single_action(action)
            self.extra_mask = []

            # early termination
            if reward == 1 or reward == -1:
                done = True

        return reward, action, done

    def mount(self, code):
        self.program.append(code)
        