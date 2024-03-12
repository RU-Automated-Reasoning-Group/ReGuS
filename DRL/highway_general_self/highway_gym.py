import highway_env
highway_env.register_highway_envs()

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from highway_env import utils

import matplotlib.pyplot as plt

import pdb


class HighwayEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, task='highway-fast-v0'):
        # required init
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # observation space for perception labels
        # (front_is_clear, left_is_clear, right_is_clear)
        # self.observation_space = spaces.Discrete(3)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(3,))

        # action space 
        # (lane_left, lane_right, faster, slower, idle)
        self.action_space = spaces.Discrete(5)

        # define environment
        self.task = task
        self.env = gym.make(task, render_mode='rgb_array')
        # self.observation_space = self.env.observation_space

        self.config = self.env.config
        # self.config['observation']['type'] = 'TimeToCollision'
        self.config['observation']['type'] = 'Kinematics'
        self.config['lanes_count'] = 8
        self.config['vehicles_density'] = 1.0
        self.env.configure(self.config)
        self.env.reset()

        self.observation_space = self.env.observation_space

        self.max_steps = 30
        # self.max_steps = 150

        self.sum_reward = 0

    def no_fuel(self):
        return not self.steps < self.max_steps

    # get observation
    def _get_obs(self):
        return self.env.observation_type.observe()

    # get information
    def _get_info(self):
        return {}

    def reset(self, seed=None):
        self.state, _ = self.env.reset(seed=seed)
        self.steps = 0
        self.sum_reward = 0

        observation = self._get_obs()

        return observation, self._get_info()

    def custom_reward(self):
        # crashed
        target_vehicle = self.env.vehicle 
        if target_vehicle.crashed:
            return -1

        # speed reward
        target_speed = target_vehicle.speed * np.cos(target_vehicle.heading)
        target_speed = utils.lmap(target_speed, self.env.config['reward_speed_range'], [0, 1])
        # try
        if target_speed <= 0.9:
            target_speed = 0

        # lane reward
        target_lane = target_vehicle.target_lane_index[2]
        neighbours = self.env.road.network.all_side_lanes(target_vehicle.lane_index)
        target_lane = target_lane / max(len(neighbours)-1, 1)

        # get all reward
        if target_lane == 1 and target_speed != 0:
            reward = 1.0
        # elif target_lane != 1 and target_speed != 0:
        #     reward = 0.8
        else:
            reward = 0.0

        return reward


    def step(self, action):
        # do action
        action = action.item()
        # lane_left, idle, lane_right, faster, slower
        state, reward, done, _, info = self.env.step(action)

        # debug
        # plt.figure()
        # plt.imshow(self.env.render())
        # plt.savefig('direct_store/{}.png'.format(action))
        # plt.close()

        self.steps += 1

        if done:
            reward = -1
        else:
            reward = self.custom_reward()

        self.state = state

        done = reward == -1 or self.steps >= self.max_steps
        
        info = self._get_info()
        observation = self._get_obs()

        if reward == -1:
            reward = min(reward, -self.sum_reward-1)
        else:
            self.sum_reward += reward

        return observation, reward, done, done, info