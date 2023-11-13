import highway_env
highway_env.register_highway_envs()

import gymnasium as gym
from gym import spaces
import numpy as np

from highway_env import utils
from highway_general.dsl import TTC_view

import pdb


class HighwayEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, task='highway-v0'):
        # required init
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # observation space for perception labels
        # (front_is_clear, left_is_clear, right_is_clear)
        self.observation_space = spaces.Discrete(3)

        # action space 
        # (lane_left, lane_right, faster, slower, idle)
        self.action_space = spaces.Discrete(5)

        # define environment
        self.task = task
        self.env = gym.make(task)
        self.config = self.env.config
        self.config['observation']['type'] = 'TimeToCollision'
        self.config['lanes_count'] = 8
        self.config['vehicles_density'] = 1.0
        self.env.configure(self.config)

        self.max_steps = 50
        # self.max_steps = 150

    # get observation
    def _get_obs(self):
        front_is_clear = TTC_view('front_is_clear', self.env)
        left_is_clear = TTC_view('left_is_clear', self.env)
        right_is_clear = TTC_view('right_is_clear', self.env)

        observation = [float(front_is_clear), 
                       float(left_is_clear), 
                       float(right_is_clear)]

        return np.array(observation)

    # get information
    def _get_info(self):
        return {}

    def reset(self):
        self.state, _ = self.env.reset()
        self.steps = 0

        observation = self._get_obs()

        return observation

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

        self.steps += 1

        if done:
            reward = -1
        else:
            reward = self.custom_reward()

        self.state = state

        done = reward == -1 or self.steps >= self.max_steps
        
        info = self._get_info()
        observation = self._get_obs()

        return observation, reward, done, info