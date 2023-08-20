import copy
from sys import prefix

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


WHILE_COND = k_cond(negation=False, cond=k_cond_without_not('front_is_clear'))
IF_COND = k_cond(negation=False, cond=k_cond_without_not('markers_present'))


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

        # for TopOff problem
        self.while_init_abs_state = True
        self.if_init_abs_state = None  # it depends

        self.if_is_done = False
        self.while_is_done = False

        self.working_while = False
        self.working_if = False

    # NOTE: try to solve the lazy predict problem
    #       if (self.if_init_abs_state) is False, predict lazily
    #       if (self.if_init_abs_state) is True, predict if_stmts first
    def step(self, action):

        done = False
        reward = 0.0
        
        # build the new line of code
        action = k_action(ACTION_NAME[action.item()])

        # which one come first
        if self.if_init_abs_state is None:
            self.if_init_abs_state = self.robot.execute_single_cond(IF_COND)

            if self.if_init_abs_state:
                self.working_if = True
            else:
                self.working_while = True

        if self.working_if:
            
            # append and execute
            self.program.append_if(action)
            r = self.robot.execute_single_action(action)
            if r == 1:
                self.program.print(prefix='[NOT REALLY][REWARD 1]')
                return self._get_obs(), 1, True, {}

            # TODO: how to terminate the if?
            # NOTE: first we assume only ONE stmt in if?
            self.if_is_done = True
            self.working_if = False

            if not self.while_is_done:
                self.working_while = True
            else:
                # execute the rest of stmts in while
                # then restart
                if not self.robot.no_fuel():
                    for s in self.program.stmts_while:
                        if not self.robot.no_fuel():
                            self.robot.execute_single_action(s)
                    reward = self.program.restart_while(self.robot)
                    done = True  # TODO: I ignored this before...
                else:
                    self.program.print(prefix='[NO FUEL][REWARD 0]')
                    return self._get_obs(), 0, True, {}

                
        elif self.working_while:

            # append and execute
            self.program.append_while(action)
            r = self.robot.execute_single_action(action)
            if r == 1:
                self.program.print(prefix='[NOT REALLY][REWARD 1]')
                return self._get_obs(), 1, True, {}

            if self.robot.execute_single_cond(WHILE_COND):
                self.while_is_done = True
                self.working_while = False

                if not self.if_is_done:
                    # try to complete the program
                    self.working_if = True

                    if self.robot.no_fuel():
                        done = True

                    # repeat stmts in while to satisfy if condition
                    while not self.robot.execute_single_cond(IF_COND) and not self.robot.no_fuel():
                        if not self.robot.no_fuel():
                            for s in self.program.stmts_while:
                                if not self.robot.no_fuel():
                                    r = self.robot.execute_single_action(s)
                                    if r == 1:
                                        self.program.print(prefix='[EARLY][REWARD 1]')
                                        return self._get_obs(), 1, True, {}
                                else:
                                    done = True
                        else:
                            done = True

                else:
                    reward = self.program.restart_while(self.robot)
                    #if reward == 1:
                    #    self.program.print(prefix='[REALLY][REWARD 1]')
                    done = True
    
        observation = self._get_obs()
        info = {}

        if done:
            self.program.print(prefix='[REWARD {}]'.format(reward))
        
        return observation, reward, done, info

    def _get_obs(self):
        robot_state = torch.stack([torch.from_numpy(self.robot.get_state())], dim=0)
        obs = self.encoder(robot_state.float()).detach().numpy()
        return obs

    def reset(self):
        self.seed += 1
        self.robot = KarelRobot(task=self.task, seed=self.seed)
        self.program = copy.deepcopy(self.init_program)

        # for TopOff problem
        self.while_init_abs_state = True
        self.if_init_abs_state = None  # it depends

        self.if_is_done = False
        self.while_is_done = False

        self.working_while = False
        self.working_if = False

        return self._get_obs()
