import os
import imageio

# NOTE: sb3 doesn't support gymnasium spaces?
# import gymnasium as gym
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from ant_program_env_utils import *
from dsl import *

import pdb

class AntProgramEnv(gym.Env):
    
    # abstract directions
    DIRECTIONS = [
        'POS_X', 'POS_Y', 'NEG_X', 'NEG_Y'
    ]

    UNIT_VECTORS = {
        'POS_X': np.array([1, 0]),
        'POS_Y': np.array([0, 1]),
        'NEG_X': np.array([-1, 0]),
        'NEG_Y': np.array([0, -1])
    }
    
    def __init__(self, 
            env, 
            env_points, 
            env_areas, 
            env_distance_ref, 
            models,
            init_direction, 
            max_episode_length,
            goal_threshold=0.5,
            goal_detection_threshold=4.0,
            distance_threshold=1.5,
            gear_size=9,
            box_size=2,
            save_trajectory=True,
            debug=False,
            seed=None,
        ):

        self.env = env
        self.env_points = env_points
        self.env_areas = env_areas
        self.env_distance_ref = env_distance_ref
        self.models = models
        self.init_direction = init_direction
        self.direction = self.init_direction
        self.max_episode_length = max_episode_length
        self.goal_threshold = goal_threshold
        self.goal_detection_threshold = goal_detection_threshold
        self.distance_threshold = distance_threshold
        self.gear_size = gear_size
        self.box_size = box_size
        self.save_trajectory = save_trajectory
        self.debug = debug
        self.seed = seed
        self.cur_goal = 1.0

        # keep track of env state
        self.goal = None
        self.steps = 0
        self.obs = None
        self.ob = None
        self.xy = None
        self.xy_distance = None
        self.sensor_info = None

        self.history_x = []
        self.history_y = []
        self.history_xy = []
        self.frames = []

        self.keep_prev_trend = False
        self.final_reach = False

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 4), dtype=np.uint8)

        self.active = True
        self.force_execution = False

        # debug
        self.debug_idx = 0


    def no_fuel(self):
        return self.steps >= self.max_episode_length

    #####################
    # utility functions
    #####################

    def reset(self):
        self.steps = 0
        self.direction = self.init_direction
        self.obs = self.env.reset()
        self.goal = self.obs['desired_goal'][:2]
        self.ob = self.obs['raw_observation']
        self.xy = self.ob[:2]
        self.xy_distance = np.inf
        self.sensor_info = self.get_sensor_info()
        self.history_x = []
        self.history_y = []
        self.active = True
        self.final_reach = False

        return self.get_abs_obs()

    def get_abs_obs(self):
        abs_obs = [
            self.front_is_clear(),
            self.left_is_clear(),
            self.right_is_clear(),
            self.present_goal(),
        ]
        return np.array(abs_obs).astype(int)

    def get_area(self, x=None, y=None):
        env_points = self.env_points
        env_areas = self.env_areas

        agent_area = None
        if x is None and y is None:
            xy = self.xy
            x, y = xy[0], xy[1]
        for area in self.env_areas:
            top_left = env_points[env_areas[area][0]]
            bottom_right = env_points[env_areas[area][1]]
            if x >= top_left[0] and y <= top_left[1] \
                and x <= bottom_right[0] and y >= bottom_right[1]:
                agent_area = area
                break
        
        return agent_area
        
    def get_sensor_info(self, offset=None):

        broken = False
        sensor_info = {
            'POS_Y': None, 
            'NEG_Y': None,
            'NEG_X': None,
            'POS_X': None,
        }

        agent_area = self.get_area()
        assert agent_area is not None, 'cannot locate the ant in given areas'

        xy = self.xy
        if offset is None:
            x, y = xy[0], xy[1]
        else:
            x, y = xy[0] + offset[0], xy[1] + offset[1]
            if not self.get_area(x, y):
                broken = True
        
        agent_area = self.get_area(x, y)
        if not agent_area:
            broken = True
        else:
            top_ref, bottom_ref, left_ref, right_ref = self.env_distance_ref[agent_area]
            sensor_info = {
                'POS_Y': top_ref - y, 
                'NEG_Y': y - bottom_ref,
                'NEG_X': x - left_ref,
                'POS_X': right_ref - x,
            }

        if broken:
            for direction in sensor_info:
                sensor_info[direction] = None

        return sensor_info

    def POS_X(self):
        return self.get_action(self.models[0], self.ob)

    def NEG_X(self):
        return self.get_action(self.models[1], self.ob)

    def POS_Y(self):
        return self.get_action(self.models[2], self.ob)

    def NEG_Y(self):
        return self.get_action(self.models[3], self.ob)

    @staticmethod
    def get_action(model, x):
        with torch.no_grad():
            x = x[2:29]
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x, deterministic=True)
            
        return torch.from_numpy(action)

    def is_goal_visible(self):
        
        pt1 = Point(self.xy[0], self.xy[1])
        pt2 = Point(self.goal[0], self.goal[1])

        visible = True
        for wall in self.env.walls:
            
            max_x, min_x = wall.max_x, wall.min_x
            max_y, min_y = wall.max_y, wall.min_y
            
            ptA = Point(min_x, max_y)
            ptB = Point(min_x, min_y)
            ptC = Point(max_x, min_y)
            ptD = Point(max_x, max_y)

            # intersect = is_intersect(pt1, pt2, ptA, ptB) and \
            #     is_intersect(pt1, pt2, ptB, ptC) and \
            #     is_intersect(pt1, pt2, ptC, ptD) and \
            #     is_intersect(pt1, pt2, ptD, ptA)
            intersect = is_intersect(pt1, pt2, ptA, ptB) or \
                is_intersect(pt1, pt2, ptB, ptC) or \
                is_intersect(pt1, pt2, ptC, ptD) or \
                is_intersect(pt1, pt2, ptD, ptA)
            
            visible = visible and not intersect
            if not visible:
                return False
                
        return True

    def check_reward(self):
        return 1 if self.present_goal() else 0

    def is_goal_reachable(self):

        visible = self.is_goal_visible()

        if visible and self.xy_distance < self.goal_detection_threshold or self.final_reach:
            omega = angle_between(self.UNIT_VECTORS['POS_X'], self.goal - self.xy)
            theta = angle_between(self.UNIT_VECTORS[self.direction], self.goal - self.xy)
            if np.abs(theta) < np.pi / 12:
                self.final_reach = True
                return True, omega, theta
            else:
                return True, omega, None

        return False, None, None
    
    def plot_trajectory(self, domain, exp_index, title=None):

        if domain in ('AntU'):
            fig, ax = plt.subplots(figsize=(6, 10))
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-10.5, 10.5)
            ax.set_xticks(range(-4, 5, 1))
            ax.set_yticks(range(-10, 11, 1))

        elif domain in ('AntFb', 'AntMaze', 'AntFg'):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-10.5, 10.5)
            ax.set_ylim(-10.5, 10.5)
            ax.set_xticks(range(-10, 11, 1))
            ax.set_yticks(range(-10, 11, 1))
            
        plt.autoscale(enable=False)
        ax.plot(self.history_x, self.history_y, color='blue', label='trajectory')
        for wall in self.env.walls:
            ax.vlines(x=wall.endpoint1[0], ymin=wall.endpoint2[1], ymax=wall.endpoint1[1])
            ax.hlines(y=wall.endpoint2[1], xmin=wall.endpoint3[0], xmax=wall.endpoint2[0])
            ax.vlines(x=wall.endpoint3[0], ymin=wall.endpoint3[1], ymax=wall.endpoint4[1])
            ax.hlines(y=wall.endpoint4[1], xmin=wall.endpoint4[0], xmax=wall.endpoint1[0])

        if title is not None:
            plt.title(title)
        plt.savefig('store_figs/[{}]-trajectory-[{}]-[{}]'.format(domain, exp_index, len(self.history_x)))
        plt.close()

    #####################
    # predicate functions
    #####################

    def execute_single_cond(self, cond):
        if 'not' in str(cond):
            cond = str(cond).split('(')[-1].split(')')[0]
            return not getattr(self, str(cond))()
        else:
            cond = str(cond)
            return getattr(self, str(cond))()
        
    def execute_single_action(self, action):
        getattr(self, str(action).strip())()
        return self.check_reward()

    def present_goal(self):
        # return self.xy_distance < self.goal_threshold

        # only for debug
        reachable, omega, theta = self.is_goal_reachable()
        return self.xy_distance < self.goal_threshold or reachable

    def front_is_clear(self):
        # if self.final_reach:
        #     return False
        return self.sensor_info[self.direction] >= self.distance_threshold['front']

    def direction_is_clear(self, abs_direction):
        # if self.final_reach:
        #     return False
        
        direction_fn = self.get_left_direction if abs_direction == 'left' else self.get_right_direction
        
        base_offset = self.box_size[0] * self.UNIT_VECTORS[direction_fn(self.direction)]
        top_offset = base_offset + self.box_size[1] * self.UNIT_VECTORS[self.direction]
        bottom_offset = base_offset + self.box_size[1] * self.UNIT_VECTORS[direction_fn(direction_fn(self.direction))]

        direction = direction_fn(self.direction)

        top_is_overlap = False
        center_is_overlap = False
        bottom_is_overlap = False

        if self.get_sensor_info(top_offset)[direction]:
            top_is_clear = self.get_sensor_info(top_offset)[direction] >= self.distance_threshold[abs_direction]
        else:
            top_is_clear = False
            top_is_overlap = True
        
        if self.get_sensor_info(base_offset)[direction]:
            center_is_clear = self.get_sensor_info(base_offset)[direction] >= self.distance_threshold[abs_direction]
        else:
            center_is_clear = False
            center_is_overlap = True
        
        if self.get_sensor_info(bottom_offset)[direction]:
            bottom_is_clear = self.get_sensor_info(bottom_offset)[direction] > self.distance_threshold[abs_direction]
        else:
            bottom_is_clear = False
            bottom_is_overlap = True

        if not bottom_is_overlap:
            is_clear = bottom_is_clear
        else:
            is_clear = not top_is_overlap and top_is_clear and \
                    not center_is_overlap and center_is_clear

        return is_clear

    def left_is_clear(self):
        return self.direction_is_clear(abs_direction='left')
    
    def right_is_clear(self):
        return self.direction_is_clear(abs_direction='right')

    #####################
    # dynamic functions
    #####################

    def move(self):

        # reachable, omega, theta = self.is_goal_reachable()

        # if self.final_reach:
        #     return self._reach(omega)

        if self.debug:
            title = 'fic:{} ric:{} lis:{}'.format(self.front_is_clear(), self.right_is_clear(), self.left_is_clear())
            self.plot_trajectory('AntFb', self.debug_idx, title='move\n'+title)
            self.debug_idx += 1
        success, done = self._move(self.direction)

        return success, done
    
    def _step(self, act):
        obs, _, _, info = self.env.step(act)
        self.obs = obs
        self.ob = self.obs['raw_observation']
        self.xy = self.ob[:2]
        self.xy_distance = info['xy-distance'].item()
        self.sensor_info = self.get_sensor_info()
        self.steps += 1
        if self.save_trajectory:
            self.history_x.append(self.xy[0])
            self.history_y.append(self.xy[1])
            self.history_xy.append(self.xy_distance)
        # if self.debug:
        #     print('[move]', self.xy, self.xy_distance, self.steps, 'front_is_clear', self.front_is_clear(), 'right_is_clear', self.right_is_clear(), 'left_is_clear', self.left_is_clear())

        # debug
        reachable, omega, theta = self.is_goal_reachable()
        success = reachable or self.xy_distance < self.goal_threshold
        # success = self.xy_distance < self.goal_threshold
        done = self.steps >= self.max_episode_length or success
        
        return success, done

    def _transit(self, index, direction, prev_direction):
        f1, f2 = index / self.num_prev_moves, 1 - index / self.num_prev_moves
        act1 = getattr(self, direction)() * self.gear_size
        act2 = getattr(self, prev_direction)() * self.gear_size
        return self._step(act1 * f1 + act2 * f2)

    def _reach(self, omega):
        f1, f2 = np.cos(omega), np.sin(omega)
        direction1 = 'POS_X' if f1 >= 0 else 'NEG_X'
        direction2 = 'POS_Y' if f2 >= 0 else 'NEG_Y'
        f1, f2 = np.abs(f1), np.abs(f2)        
        act1 = getattr(self, direction1)() * self.gear_size
        act2 = getattr(self, direction2)() * self.gear_size
        return self._step(act1 * f1 / (f1 + f2) + act2 * f2 / (f1 + f2))

    def _move(self, direction):
        act = getattr(self, direction)() * self.gear_size
        return self._step(act)

    def get_left_direction(self, direction=None):
        DIRECTIONS = self.DIRECTIONS
        direction = self.direction if direction is None else direction
        index = (DIRECTIONS.index(direction) + 1) % len(DIRECTIONS)
        return DIRECTIONS[index]

    def get_right_direction(self, direction=None):
        DIRECTIONS = self.DIRECTIONS
        direction = self.direction if direction is None else direction
        index = ((DIRECTIONS.index(direction) - 1) + len(DIRECTIONS)) % len(DIRECTIONS)
        return DIRECTIONS[index]
    
    def turn_left(self):
        # if not self.final_reach:
        #     if self.debug:
        #         print('[turn_left]')
        #     self.direction = self.get_left_direction(self.direction)
        if self.debug:
            title = 'fic:{} ric:{} lis:{}'.format(self.front_is_clear(), self.right_is_clear(), self.left_is_clear())
            self.plot_trajectory('AntFb', self.debug_idx, title='turn_left\n'+title)
            self.debug_idx += 1
        self.direction = self.get_left_direction(self.direction)

    def turn_right(self):
        # if not self.final_reach:
        #     if self.debug:
        #         print('[turn_right]')
        #     self.direction = self.get_right_direction(self.direction)
        if self.debug:
            title = 'fic:{} ric:{} lis:{}'.format(self.front_is_clear(), self.right_is_clear(), self.left_is_clear())
            self.plot_trajectory('AntFb', self.debug_idx, title='turn_right\n'+title)
            self.debug_idx += 1
        self.direction = self.get_right_direction(self.direction)
