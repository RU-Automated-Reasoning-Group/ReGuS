from karel.karel import Karel
from karel.generator import KarelStateGenerator
from karel.checker import *

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

import pdb

class KarelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, task='seeder', obs_type='abs'):
        # required init
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # observation space for perception labels
        # (front_is_clear, left_is_clear, right_is_clear, markers_present)
        # self.observation_space = spaces.Discrete(4)
        if obs_type == 'abs':
            self.observation_space = spaces.Box(low=0, high=1, shape=(4,))
        elif obs_type == 'raw':
            if task == 'harvester':
                self.observation_space = spaces.Box(low=0, high=10, shape=(16,12,12), dtype=np.uint8)
            elif task == 'cleanHouse':
                self.observation_space = spaces.Box(low=0, high=10, shape=(16,14,22), dtype=np.uint8)
            elif task == 'stairClimber':
                self.observation_space = spaces.Box(low=0, high=10, shape=(16,12,12), dtype=np.uint8)
            elif task == 'topOff':
                self.observation_space = spaces.Box(low=0, high=10, shape=(16,8,8), dtype=np.uint8)
            elif task == 'fourCorners':
                self.observation_space = spaces.Box(low=0, high=10, shape=(16,12,12), dtype=np.uint8)
            else:
                self.observation_space = spaces.Box(low=0, high=10, shape=(16,8,8), dtype=np.uint8)


        # action space 
        # (move, turn_left, turn_right, put_marker, pick_marker)
        self.action_space = spaces.Discrete(5)

        # define environment
        self.task = task
        self.obs_type = obs_type
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
        elif task == 'cleanHouse':
            checker = CleanHouseChecker(self.init_state)
        elif task == 'randomMaze':
            checker = RandomMazeChecker(self.init_state)
        elif task == 'stairClimber':
            checker = SparseStairClimberChecker(self.init_state)
        elif task == 'topOff':
            checker = TopOffChecker(self.init_state)
        elif task == 'fourCorners':
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
        elif task == 'cleanHouse':
            gen_function = gen_function_base + 'clean_house'
        elif task == 'randomMaze':
            gen_function = gen_function_base + 'random_maze'
        elif task == 'stairClimber':
            gen_function = gen_function_base + 'stair_climber'
        elif task == 'topOff':
            gen_function = gen_function_base + 'top_off'
        elif task == 'fourCorners':
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
        if self.obs_type == 'abs':
            front_is_clear = self.karel.front_is_clear()
            left_is_clear = self.karel.left_is_clear()
            right_is_clear = self.karel.right_is_clear()
            markers_present = self.karel.markers_present()

            observation = [float(front_is_clear), 
                        float(left_is_clear), 
                        float(right_is_clear),
                        float(markers_present)]
            
            observation = np.array(observation)

        elif self.obs_type == 'raw':
            observation = self.karel.state
            observation = observation.transpose(2,0,1)

        else:
            raise NotImplementedError
        
        return observation

    # get information
    def _get_info(self):
        return {}

    def reset(self, seed=None):
        if seed is not None:
            self.gen = KarelStateGenerator(seed=seed)

        self.init_state = self.state_init(self.gen, self.task)
        self.checker = self.checker_init(self.task)
        self.karel = Karel(state=self.init_state.astype(int))
        self.steps = 0
        if self.task == 'doorkey':
            self.no_break = True

        observation = self._get_obs()

        return observation, self._get_info()

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
            reward = self.checker(self.karel.state)
            if self.no_break and reward > 0:
                self.break_wall()
                self.no_break = False
            done = reward == 1 or reward == -1 or self.steps >= self.max_steps
        elif self.task == 'cleanHouse':
            reward = self.checker(self.karel.state)
            done = self.checker.prev_reward == 1 or reward == -1 or self.steps >= self.max_steps
        else:
            reward = self.checker(self.karel.state)
            done = reward == 1 or reward == -1 or self.steps >= self.max_steps
        
        info = self._get_info()
        observation = self._get_obs()

        return observation, reward, done, done, info