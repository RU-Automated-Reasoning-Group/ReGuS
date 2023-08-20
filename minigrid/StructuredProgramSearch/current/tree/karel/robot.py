
# import torch

from .checker import *
from .dsl import *
from .generator import KarelStateGenerator
from .karel import Karel


# single karel robot
class KarelRobot:
    def __init__(self, task, seed=999):
        
        self.task = task
        self.seed = seed
        self.gen = KarelStateGenerator(seed=self.seed)

        # init state
        self.init_state = self.state_init(self.gen, self.task)
        self.karel = Karel(state=self.init_state.astype(int))

        # init checker
        self.checker = self.checker_init(self.task)

        # TODO: max_steps is pre-set now
        self.steps = 0
        self.action_steps = 0
        # self.d
        # self.max_steps = 100
        self.max_steps = 1000

        self.active = True  # used for soft termination
        self.force_execution = False  # without considering abs_state

        # TODO: only used for determine termination of WHILE
        self.while_start_robot_pos = None
        self.while_moved = False

    def no_fuel(self):

        return not self.steps <= self.max_steps

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
        elif task == "topOffPick":
            gen_function = gen_function_base + "top_off_pick"
        else:
            print('Please check the task')
            exit()

        state, _, _, _, _ = getattr(gen, gen_function)()
        
        # NOTE: easier for printing
        return state

    def checker_init(self, task):

        if task == "cleanHouse":
            # checker = CleanHouseChecker(self.init_state)
            checker = CleanHouseCheckerAdd(self.init_state)
        elif task == "harvester":
            checker = HarvesterChecker(self.init_state)
        elif task == "randomMaze":
            #checker = RandomMazeChecker(self.init_state)
            checker = AccRandomMazeChecker(self.init_state)
        elif task == "fourCorners":
            checker = FourCornerChecker(self.init_state)
        elif task == "stairClimber":
            checker = StairClimberChecker(self.init_state)
            # checker = AccStairClimberChecker(self.init_state)
        elif task == "topOff":
            checker = TopOffChecker(self.init_state)
        elif task == 'topOffPick':
            checker = TopOffPickChecker(self.init_state)

        return checker

    def get_state(self, printing=False):
        return self.karel.state if not printing else self.karel.state.astype(bool)

    def execute_single_action(self, action):
        assert isinstance(action, k_action)
        
        # return reward
        return action(self.karel, self)

    def execute_single_cond(self, cond):
        assert isinstance(cond, k_cond) 

        return cond(self.karel)

    def check_reward(self):
        return self.checker(self.get_state())

    # def get_search_reward(self):


    def draw(self, *args, **kwargs):
        return self.karel.draw(*args, **kwargs)
