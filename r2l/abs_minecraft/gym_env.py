from abs_minecraft.robot import MinecraftRobot
from abs_minecraft.environment import mine_env

import gym
from gym import spaces
import pygame
import numpy as np

class MinecraftEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        # required init
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # observation space for perception labels
        # (front_is_clear, left_is_clear, right_is_clear, 
        #  front_is_door, left_is_goal, right_is_goal)
        self.observation_space = spaces.Discrete(6)

        # action space 
        # (move, turn_left, turn_right, use)
        self.action_space = spaces.Discrete(4)

        self.prev_reward = 0.0

    # get observation
    def _get_obs(self):
        observation = [self.env.execute_single_cond(cond) for cond in self.env.valid_perception]

        return np.array(observation)

    # get information
    def _get_info(self):
        return {}

    def reset(self):
        self.env = MinecraftRobot(seed=None)

        observation = self._get_obs()

        return observation

    def step(self, action):
        # do action
        reward = self.env.execute_single_action(self.env.valid_actions[int(action)])
        self.env.steps += 1

        done = not self.env.active or self.env.no_fuel()

        info = self._get_info()
        observation = self._get_obs()

        # try reward
        if reward != self.prev_reward:
            get_reward = reward - self.prev_reward
            self.prev_reward = reward
            reward = get_reward
        else:
            reward = 0

        return observation, reward, done, info