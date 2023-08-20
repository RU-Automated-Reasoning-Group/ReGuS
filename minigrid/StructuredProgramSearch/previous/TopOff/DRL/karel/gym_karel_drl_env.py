import copy

import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn

from karel.robot import KarelRobot
from karel.dsl import k_action, k_cond, k_cond_without_not
#from karel.program import AbsState, Node, Branch, Program
from karel.program import *


ACTION_INDEX = [0, 1, 2, 3, 4]
ACTION_NAME = [
    'move',
    'turn_right',
    'turn_left',
    'pick_marker',
    'put_marker'
]


COND_DICT = {
    'front_is_clear'    : k_cond(negation=False, cond=k_cond_without_not('front_is_clear')),
    'left_is_clear'     : k_cond(negation=False, cond=k_cond_without_not('left_is_clear')),
    'right_is_clear'    : k_cond(negation=False, cond=k_cond_without_not('right_is_clear')),
    'no_markers_present': k_cond(negation=False, cond=k_cond_without_not('no_markers_present')),
}


class KarelGymDRLEnv(gym.Env):

    def __init__(self, task, seed, encoder):
        super().__init__()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float)

        self.task = task
        self.seed = seed
        self.encoder = encoder

        self.robot = KarelRobot(task=self.task, seed=self.seed)
        self.init_robot = copy.deepcopy(self.robot)  # to backup
        
    def step(self, action, final_step):

        done = False
        reward = 0  # -0.01

        # for quick access
        robot = self.robot
        action = k_action(ACTION_NAME[action.item()])
        print(action)

        if final_step:
            r = robot.execute_single_action(k_action('null'))

            print('okay, final step')
            return self.early_return(r)

        if not robot.no_fuel():
            r = robot.execute_single_action(action)
            if r == 1:
                return self.early_return(1)

        observation = self._get_obs()
        info = {}

        return observation, reward, done, info

    def early_return(self, reward, meta_info=None):
        if meta_info:
            print(meta_info)
        return self._get_obs(), reward, True, {}

    def _get_obs(self):
        robot_state = torch.stack([torch.from_numpy(self.robot.get_state())], dim=0)
        obs = self.encoder(robot_state.float()).detach().numpy()
        return obs

    def reset(self):
        self.seed += 1
        self.robot = KarelRobot(task=self.task, seed=self.seed)
        self.init_robot = copy.deepcopy(self.robot)  # to backup
        return self._get_obs()

