import highway_env
highway_env.register_highway_envs()

import gymnasium as gym
import numpy as np

from highway_env import utils
# from .dsl import *

import pdb

class HighwayRobot:
    def __init__(self, task='highway-v0', seed=999, config_set=None, view='TTC'):
        # init environment
        self.task = task
        self.seed = seed
        self.env = gym.make(task)

        self.config = self.env.config
        if config_set is None:
            self.config['observation']['type'] = 'TimeToCollision'
            # self.config['observation']['horizon'] = 100
            self.config['lanes_count'] = 8
            # self.config['vehicles_density'] = 1.0
            self.config['vehicles_density'] = 0.2
        else:
            for k in config_set:
                self.config[k] = config_set[k]
        self.env.configure(self.config)
        self.state, _ = self.env.reset(seed=seed)
        
        # init reward
        _, reward, _, _, _ = self.env.step(1)
        self.cur_reward = reward
        self.past_reward = []
        # self.debug_reward = []
        self.state, _ = self.env.reset(seed=seed)

        # other init for control
        self.steps = 0
        self.action_steps = 1
        # self.max_steps = 150
        self.max_steps = 40
        # self.max_steps = 80
        # self.max_steps = 25
        # self.max_steps = 15

        self.reward_steps = 1

        self.view = view
        if self.view == 'Grid':
            self.ego_grid = [4, 5]

        self.active = True
        self.force_execution = False

    def reset(self):
        seed = self.seed
        self.state, _ = self.env.reset(seed=seed)
        
        # init reward
        _, reward, _, _, _ = self.env.step(1)
        self.cur_reward = reward
        self.past_reward = []
        # self.debug_reward = []
        self.state, _ = self.env.reset(seed=seed)

        # other init for control
        self.steps = 0
        self.action_steps = 1
        self.reward_steps = 1

        self.active = True
        self.force_execution = False

    def no_fuel(self):

        return not self.action_steps <= self.max_steps
        # return not self.steps <= self.max_steps

    def get_state(self):
        
        return self.env.observation_type.observe()
    
    def execute_single_action(self, action):
        # if self.action_steps >= 74:
        #     pdb.set_trace()

        # assert isinstance(action, h_action)

        state, reward = action(self.env, self)
        if reward == -1:
            assert self.custom_reward() == 0
        reward = self.custom_reward()

        self.state = state
        self.cur_reward += reward
        if 'roundabout' in self.task and reward != 0:
            self.reward_steps += 1
        elif 'roundabout' not in self.task:
            self.reward_steps += 1

        if self.env.vehicle.crashed:
            self.cur_reward = -1

        return self.check_reward()
    
    def execute_single_cond(self, cond):
        # assert isinstance(cond, h_cond)

        return cond(self.env, self.view)

    # def custom_reward(self):
    #     # crashed
    #     target_vehicle = self.env.vehicle 
    #     if target_vehicle.crashed:
    #         return 0
        
    #     # not count for first 3 steps
    #     if self.action_steps <= 3:
    #         self.cur_reward = 0
    #         return 0

    #     # speed reward
    #     target_speed = target_vehicle.speed * np.cos(target_vehicle.heading)
    #     target_speed = utils.lmap(target_speed, self.env.config['reward_speed_range'], [0, 1])
    #     # try
    #     if target_speed <= 0.9:
    #         target_speed = 0

    #     # lane reward
    #     target_lane = target_vehicle.target_lane_index[2]
    #     neighbours = self.env.road.network.all_side_lanes(target_vehicle.lane_index)
    #     target_lane = target_lane / max(len(neighbours)-1, 1)

    #     # get all reward
    #     # reward = 0.8 * target_speed + 0.2 * target_lane
    #     if target_speed == 0:
    #         reward = 0
    #     else:
    #         reward = 0.8 * target_speed + 0.2 * target_lane

    #     return reward

    def custom_reward(self):
        # crashed
        target_vehicle = self.env.vehicle 
        if target_vehicle.crashed:
            return 0

        # not count for first 3 steps
        if self.action_steps <= 3:
            self.cur_reward = 0
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
        # reward = 0.8 * target_speed + 0.2 * target_lane
        if target_lane == 1 and target_speed != 0:
            reward = 1.0
        elif target_lane != 1 and target_speed != 0:
            reward = 0.8
        else:
            reward = 0.0

        return reward


    def check_reward(self):
        if 'roundabout' not in self.task:
            assert self.reward_steps == self.action_steps

        if self.cur_reward != -1:
            # return self.cur_reward / self.action_steps
            # return self.cur_reward
            return self.cur_reward / self.reward_steps
        else:
            return -1
        
    def check_eval_reward(self):
        if 'roundabout' not in self.task:
            assert self.reward_steps == self.action_steps
        
        cur_reward = self.cur_reward
        if cur_reward == -1:
            return -1
        
        # cur_n = self.action_steps
        cur_n = self.reward_steps
        for p_r, p_n in self.past_reward:
            cur_reward += p_r * p_n
            cur_n += p_n

        # reward = cur_reward / cur_n  + 0.01 * float(len(self.past_reward)>0)
        reward = cur_reward / cur_n

        return reward