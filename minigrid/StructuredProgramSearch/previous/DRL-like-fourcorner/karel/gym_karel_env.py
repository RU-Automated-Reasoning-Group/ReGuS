import copy

import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn

from .robot import KarelRobot
from .dsl import k_action, k_cond, k_cond_without_not


ACTION_INDEX = [0, 1, 2, 3, 4]
ACTION_NAME = [
    'move',
    'turn_right',
    'turn_left',
    'pick_marker',
    'put_marker'
]


INNER_COND = k_cond(negation=False, cond=k_cond_without_not('front_is_clear'))
OUTER_COND = k_cond(negation=False, cond=k_cond_without_not('no_markers_present'))


class KarelGymEnv(gym.Env):

    def __init__(self, task, program, seed, encoder):
        super().__init__()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float)

        self.task = task
        self.seed = seed
        self.encoder = encoder

        self.init_program = program
        self.program = program
        self.robot = KarelRobot(task=self.task, seed=self.seed)

        self.inner_init_abs_state = True
        self.outer_init_abs_state = True

        self.inner_is_done = False
        self.outer_is_done = False

    def step(self, action):

        done = False
        reward = 0.0
        
        # build the new line of code
        action = k_action(ACTION_NAME[action.item()])
        
        # execute the new line of code
        self.robot.execute_single_action(action)

        # check inner loop
        if not self.inner_is_done:

            # fill the inner loop
            self.program.append_inner(action)

            if self.robot.execute_single_cond(INNER_COND) == self.inner_init_abs_state:

                # restart the inner loop
                self.program.restart_inner(self.robot)
                if self.robot.no_fuel():
                    done = True
                    self.program.print('[NO FUEL INNER]')
                else:
                    self.inner_is_done = True

        # check outer loop
        elif not self.outer_is_done:

            # fill the outer loop
            self.program.append_outer(action)

            if self.robot.execute_single_cond(OUTER_COND) == self.outer_init_abs_state:
                
                # restart the outer loop
                reward = self.program.restart_outer(self.robot)
                if self.robot.no_fuel():
                    done = True
                    self.program.print('[NO FUEL OUTER]')
                else:
                    self.outer_is_done = True
        
        if self.inner_is_done and self.outer_is_done:
            done = True
            self.program.print('[DONE         ]')

        observation = self._get_obs()
        info = {}

        return observation, reward, done, info

    def _get_obs(self):
        robot_state = torch.stack([torch.from_numpy(self.robot.get_state())], dim=0)
        obs = self.encoder(robot_state.float()).detach().numpy()
        return obs

    def reset(self):
        self.seed += 1
        self.robot = KarelRobot(task=self.task, seed=self.seed)
        self.program = copy.deepcopy(self.init_program)

        self.inner_init_abs_state = True
        self.outer_init_abs_state = True

        self.inner_is_done = False
        self.outer_is_done = False

        return self._get_obs()
