import highway_env
highway_env.register_highway_envs()

import gymnasium as gym
import numpy as np

from highway_env import utils
from .dsl import *

import pdb
from utils.logging import log_and_print

class HighwayRobot(gym.Wrapper):
    def __init__(self, task='highway-fast-v0', seed=999, config_set=None, act_dim=5, view='TTC', eval=False):
        # init environment
        self.task = task
        self.seed = seed
        self.env = gym.make(task)
        self.eval = eval
        super().__init__(self.env)

        self.config = self.env.config
        if config_set is None:
            self.config['observation']['type'] = 'TimeToCollision'
            self.config['lanes_count'] = 8
            self.config['vehicles_density'] = 1.0
        else:
            for k in config_set:
                self.config[k] = config_set[k]
        self.env.configure(self.config)
        self.env.reset(seed=seed)

        # environment space
        self.action_space = gym.spaces.Discrete(act_dim)
        self.observation_space = self.env.observation_space
        self.perc_labels = [h_cond_without_not('left_is_clear'), 
                            h_cond_without_not('right_is_clear'), 
                            h_cond_without_not('front_is_clear')]

        # init reward
        self.cur_reward = 0
        self.state = None

        # other init for control
        self.steps = 0
        self.action_steps = 1
        self.max_steps = 30
        self.reward_steps = 1

        self.view = view
        if self.view == 'Grid':
            self.ego_grid = [4, 5]

        self.info = None
        self.active = True
        self.force_execution = False

    def reset(self, seed=None, options=None):
        # reset reward
        self.cur_reward = 0
        self.state = None

        # other init for control
        self.steps = 0
        self.action_steps = 1
        self.reward_steps = 1

        self.active = True
        self.force_execution = False

        # do reset
        if seed is not None:
            if options is not None:
                self.env.reset(seed=seed, options=options)
            self.env.reset(seed=seed)
        else:
            self.env.reset()

        return self._get_obs()

    def no_fuel(self):

        return not self.action_steps <= self.max_steps

    def get_state(self):

        return self.env.observation_type.observe()

    def _get_obs(self):
        observation = []
        for label in self.perc_labels:
            observation.append(float(self.execute_single_cond(label)))

        return np.array(observation)

    def execute_single_action(self, action):
        # assert isinstance(action, h_action)

        state, reward = action(self.env, self)
        if self.eval:
            reward = reward
        else:
            if reward == -1:
                assert self.custom_reward() == 0
            reward = self.custom_reward()

        self.state = state
        self.cur_reward += reward
        if 'roundabout' in self.task and reward != 0:
            self.reward_steps += 1
        elif 'roundabout' not in self.task:
            self.reward_steps += 1

        if not self.eval:
            if self.env.vehicle.crashed:
                self.cur_reward = -1
        else:
            if self.env.vehicle.crashed:
                self.active = False

        return self.check_reward()

    def execute_single_cond(self, cond):
        # assert isinstance(cond, h_cond)

        return cond(self.env, self.view)

    def custom_reward(self):
        # crashed
        target_vehicle = self.env.vehicle
        if target_vehicle.crashed:
            return 0

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
        else:
            reward = 0.0
        return reward


    def check_reward(self):
        if 'roundabout' not in self.task:
            assert self.reward_steps == self.action_steps

        if self.cur_reward != -1:
            return self.cur_reward
        else:
            return -1

    def reward(self):
        return self.check_reward()

    def step(self, act):
        # single step
        h_act = h_action(act)
        r = self.execute_single_action(h_act)
        # other
        obs = self._get_obs()
        done = r==-1
        
        pdb.set_trace()

        return obs, r, done, self.no_fuel(), self.info