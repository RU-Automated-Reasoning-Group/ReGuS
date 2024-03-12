from karel.karel import Karel
from karel.generator import KarelStateGenerator
from karel.checker import *

import gym
from gym import spaces
import pygame
import numpy as np

class KarelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, task='seeder'):
        # required init
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # observation space for perception labels
        # (front_is_clear, left_is_clear, right_is_clear, markers_present)
        self.observation_space = spaces.Discrete(4)

        # action space 
        # (move, turn_left, turn_right, put_marker, pick_marker)
        self.action_space = spaces.Discrete(5)

        # define environment
        self.task = task
        self.gen = KarelStateGenerator()
        self.max_steps = 1000
        if self.task == 'doorkey':
            self.no_break = True

    def checker_init(self, task):
        if task == 'seeder':
            checker = SeederChecker(self.init_state)
        elif task == 'doorkey':
            checker = DoorKeyChecker(self.init_state)
        elif task == 'harvester':
            checker = HarvesterChecker(self.init_state)
        elif task == 'cleanhouse':
            checker = CleanHouseChecker(self.init_state)
        elif task == 'randommaze':
            checker = RandomMazeChecker(self.init_state)
        elif task == 'stairclimber':
            checker = SparseStairClimberChecker(self.init_state)
        elif task == 'topoff':
            checker = TopOffChecker(self.init_state)
        elif task == 'fourcorners':
            checker = FourCornerChecker(self.init_state)

        return checker

    def state_init(self, gen, task):
    
        gen_function_base = "generate_single_state_"

        if task == 'seeder':
            gen_function = gen_function_base + "seeder"
        elif task == 'doorkey':
            gen_function = gen_function_base + 'doorkey'
        elif task == 'harvester':
            gen_function = gen_function_base + 'harvester'
        elif task == 'cleanhouse':
            gen_function = gen_function_base + 'clean_house'
        elif task == 'randommaze':
            gen_function = gen_function_base + 'random_maze'
        elif task == 'stairclimber':
            gen_function = gen_function_base + 'stair_climber'
        elif task == 'topoff':
            gen_function = gen_function_base + 'top_off'
        elif task == 'fourcorners':
            gen_function = gen_function_base + 'four_corners'
        else:
            print('Please check the task')
            exit()

        state, _, _, _, _ = getattr(gen, gen_function)()
        
        # NOTE: easier for printing
        return state

    # specific for door key (hard code)
    def break_wall(self):
        state = self.karel.state
        state[2, 4, 4] = 0
        state[3, 4, 4] = 0
        self.karel = Karel(state=state.astype(int))

    # get observation
    def _get_obs(self):
        front_is_clear = self.karel.front_is_clear()
        left_is_clear = self.karel.left_is_clear()
        right_is_clear = self.karel.right_is_clear()
        markers_present = self.karel.markers_present()

        observation = [float(front_is_clear), 
                       float(left_is_clear), 
                       float(right_is_clear),
                       float(markers_present)]

        return np.array(observation)

    # get information
    def _get_info(self):
        return {}

    def reset(self):
        self.init_state = self.state_init(self.gen, self.task)
        self.checker = self.checker_init(self.task)
        self.karel = Karel(state=self.init_state.astype(int))
        self.steps = 0
        if self.task == 'doorkey':
            self.no_break = True

        observation = self._get_obs()

        return observation

    def step(self, action):
        # do action
        action = action.item()
        # move
        if action == 0:
            self.karel.move()
        # turn left
        elif action == 1:
            self.karel.turn_left()
        # turn right
        elif action == 2:
            self.karel.turn_right()
        # put marker
        elif action == 3:
            # self.karel.put_marker()
            self.karel.pick_marker()
        # pick marker
        elif action == 4:
            # self.karel.pick_marker()
            self.karel.put_marker()
        # otherwise
        else:
            raise NotImplementedError
        
        self.steps += 1

        if self.task == 'seeder':
            reward, end = self.checker(self.karel.state)
            done = reward == 1 or reward == -1 or self.steps >= self.max_steps or end
        elif self.task == 'doorkey':
            reward, end = self.checker(self.karel.state)
            if self.no_break and reward > 0:
                self.break_wall()
                self.no_break = False
            done = reward == 1 or reward == -1 or self.steps >= self.max_steps or end
        elif self.task == 'cleanhouse' or self.task == 'topoff' or self.task == 'fourcorners':
            reward = self.checker(self.karel.state)
            done = self.checker.prev_reward == 1 or reward == -1 or self.steps >= self.max_steps
        else:
            reward = self.checker(self.karel.state)
            done = reward == 1 or reward == -1 or self.steps >= self.max_steps

        info = self._get_info()
        observation = self._get_obs()

        return observation, reward, done, info