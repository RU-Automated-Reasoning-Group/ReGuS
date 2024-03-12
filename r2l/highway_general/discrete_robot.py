import highway_env
highway_env.register_highway_envs()

import gymnasium as gym
import numpy as np

import os
import pdb

from .discrete_dsl import *

class HighwayRobot:
    def __init__(self, task='highway-v0', seed=999, debug=False):
        # init environment
        self.task = task
        self.env = gym.make(task)
        
        self.config = self.env.config
        self.config['observation']['type'] = 'NewTimeToCollision'
        self.config['observation']['horizon'] = 150
        self.config['lanes_count'] = 8
        # self.config['reward_speed_range'] = [15, 40]
        self.env.configure(self.config)
        self.state, _ = self.env.reset(seed=seed)

        self.lane_domain = [l_id for l_id in range(len(self.state[0]))]
        self.velocity_domain = [v_id for v_id in range(len(self.state))]

        # hard code
        self.real_velocity_domain = [20, 25, 30]
        self.time_length = 150

        # init agent information
        self.velocity = self.velocity_domain[len(self.velocity_domain) // 2]
        self.lane_pos = self.lane_domain[len(self.lane_domain) // 2]
        # self.all_time_pos = [0 for _ in self.real_velocity_domain]

        # get information from vehicles
        self._get_all_state()

        # init reward
        self.cur_reward = self.cal_cur_reward()

        # other init for control
        self.steps = 0
        self.action_steps = 1
        self.max_steps = self.time_length

        self.active = True
        self.force_execution = False
        self.debug = debug

    def _get_all_state(self):
        # vehicles are sorted by position
        vehicles = self.env.road.vehicles
        self.ttc_vehicle_map = {v_id:{l_id:[] for l_id in self.lane_domain} for v_id in self.velocity_domain}
        state_match = np.zeros_like(self.state)

        # store map
        cur_pos_x, cur_pos_y = vehicles[0].position
        for v_id, v in enumerate(vehicles[1:]):
            v_speed = v.speed
            v_pos_x, v_pos_y = v.position
            v_pos_y = int((v_pos_y - cur_pos_y) // 4 + self.lane_pos)

            # in TTC
            if v_pos_y < 0 or v_pos_y > len(self.lane_domain)-1:
                continue

            for v_id in self.velocity_domain:
                # never crash
                if v_speed > self.real_velocity_domain[v_id]:
                    continue

                # get all obstacles
                crash_time = int((v_pos_x - cur_pos_x) / (self.real_velocity_domain[v_id] - v_speed))
                assert crash_time >= 0
                if crash_time >= self.time_length:
                    continue

                # get left obstacles
                for t in range(crash_time, -1, -1):
                    # check obstacle exist
                    if self.state[v_id, v_pos_y, t] == 0:
                        break
                    # check duplicate
                    if state_match[v_id, v_pos_y, t] == 1:
                        break
                    else:
                        state_match[v_id, v_pos_y, t] = 1
                    self.ttc_vehicle_map[v_id][v_pos_y].append([v_speed, float(t)])

                # get right obstacles
                for t in range(crash_time+1, self.time_length):
                    # check obstacle exist
                    if self.state[v_id, v_pos_y, t] == 0:
                        break
                    # check duplicate
                    if state_match[v_id, v_pos_y, t] == 1:
                        break
                    else:
                        state_match[v_id, v_pos_y, t] = 1
                    self.ttc_vehicle_map[v_id][v_pos_y].append([v_speed, float(t)])

        # refresh state
        if cur_pos_y >= 4 * (self.config['lanes_count']-2):
            state_match[:, -1] = 1
        elif cur_pos_y <= 4 * 1:
            state_match[:, 0] = 1

        self.state = state_match


    def no_fuel(self):

        return not self.action_steps <= self.max_steps
        # return not self.steps <= self.max_steps
        # return not self.all_time_pos[0] < self.time_length

    def get_state(self):
        
        return self.env.observation_type.observe()
    
    # TODO: update TTC after action

    def execute_single_action(self, action):
        assert isinstance(action, h_action)

        state, reward = action(self)
        # self.state = state
        self.cur_reward += reward
        if reward == -1:
            self.cur_reward = -1

        # debug
        if self.debug:
            world = self.draw_state()
            with open('store/highway_discrete_log/highway_discrete_world.txt', 'a') as f:
                f.write(str(action))
                f.write('\n'+world)

        return self.check_reward()
    
    def execute_single_cond(self, cond):
        assert isinstance(cond, h_cond)

        return cond(self)

    # update TTC for an action
    def make_move(self):
        # update time for each vehicle
        for v_id in self.velocity_domain:
            for l_id in self.lane_domain:
                vehicles = self.ttc_vehicle_map[v_id][l_id]
                for vehicle_id in range(len(vehicles)):
                    if v_id == self.velocity:
                        self.ttc_vehicle_map[v_id][l_id][vehicle_id][1] = vehicles[vehicle_id][1] - 1
                    else:
                        delta_vel = self.real_velocity_domain[v_id] - self.real_velocity_domain[self.velocity]
                        delta_time = 1 - delta_vel / (self.real_velocity_domain[v_id] - vehicles[vehicle_id][0])
                        assert self.real_velocity_domain[v_id] > vehicles[vehicle_id][0]
                        # if vehicles[vehicle_id][1] - delta_time < 0:
                        #     pdb.set_trace()
                        self.ttc_vehicle_map[v_id][l_id][vehicle_id][1] = vehicles[vehicle_id][1] - delta_time

        # update TTC
        new_state = np.zeros_like(self.state)
        for v_id in self.velocity_domain:
            for l_id in self.lane_domain:
                for vehicle in self.ttc_vehicle_map[v_id][l_id]:
                    if vehicle[1] < self.time_length and vehicle[1] >= 0:
                        new_state[v_id, l_id, int(vehicle[1])] = 1
        if np.mean(self.state[:, 0]) == 1:
            new_state[:, 0] = 1
        elif np.mean(self.state[:, -1]) == 1:
            new_state[:, -1] = 1
        self.state = new_state

    def cal_cur_reward(self):
        # time_pos = np.ceil(self.all_time_pos[self.velocity]).astype(int)
        time_pos = 0
        coll_term = -1 * float(self.state[self.velocity, self.lane_pos, time_pos] > 0)
        lane_term = 0.1 * (self.lane_pos - self.lane_domain[0]) / (self.lane_domain[-1] - self.lane_domain[0])
        speed_term = 0.4 * (self.velocity - self.velocity_domain[0]) / (self.velocity_domain[-1] - self.velocity_domain[0])

        if coll_term == -1:
            reward = -1
        else:
            reward = coll_term + lane_term + speed_term
            reward = (reward + 1) / (0.5 + 1)

        return reward

    def check_reward(self):
        if self.cur_reward != -1:
            return self.cur_reward / self.action_steps
        else:
            return -1
        
    def draw_state(self):
        # time_pos = np.ceil(self.all_time_pos[self.velocity]).astype(int)
        time_pos = 0
        empty_char = '_'
        vehicle_char = '>'
        block_char = 'B'
        world = ''
        # draw world and ego vehicle
        for speed_id, speed in enumerate(self.state):
            for lane_id, lane in enumerate(speed):
                for col_id, pos in enumerate(lane):
                    if self.velocity == speed_id and self.lane_pos == lane_id and time_pos == col_id:
                        world += vehicle_char
                    elif pos == 0:
                        world += empty_char
                    else:
                        world += block_char
                world += '\n'
            world += '\n'

        return world
