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


OUTER_COND = k_cond(negation=False, cond=k_cond_without_not('front_is_clear'))
INNER_COND = k_cond(negation=False, cond=k_cond_without_not('no_markers_present'))


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
        self.inner_init_abs_state = True  # no_markers_presents, it depends
        self.outer_init_abs_state = True  # front_is_clear is always True
        
        # not sure, can be either True/False
        # TODO: check TofOff generator later

        # NOTE: we may need two boolean to keep track of
        #       the prediction and completion?

        self.inner_is_done = False
        self.outer_is_done = False

        self.predicting_inner = False
        self.predicting_outer = False

    # NOTE: first time try to solve the lazy predict problem 
    def step(self, action):

        done = False
        reward = 0.0
        
        # build the new line of code
        action = k_action(ACTION_NAME[action.item()])

        if not self.predicting_inner and not self.predicting_outer:
            
            if self.robot.execute_single_cond(INNER_COND):
                self.predicting_inner = True
            else:
                self.predicting_outer = True

        # check inner loop
        if self.predicting_inner:
    
            # append and execute
            self.program.append_inner(action)
            self.robot.execute_single_action(action)

            # if complete, restart and continue to predict outer
            if self.robot.execute_single_cond(INNER_COND):
                
                finished = self.program.restart_inner(self.robot)
                if finished:
                    observation = self._get_obs()
                    reward = 1
                    done = True
                    info = {}
                    return observation, reward, done, info

                self.predicting_inner = False
                self.predicting_outer = True

                self.inner_is_done = True

                # TODO: try to make this more general
                if self.outer_is_done:
                    
                    for s in self.program.stmts_outer:
                        if not self.robot.no_fuel():
                            reward = self.robot.execute_single_action(s)
                    
                    if not self.robot.no_fuel():
                        reward = self.program.restart_outer(self.robot)
                    
                    done = True

        # NOTE: things are more complex when predicting lazily
        # check outer loop
        elif self.predicting_outer:

            # append and execute
            self.program.append_outer(action)
            self.robot.execute_single_action(action)

            # if complete, try to restart
            if self.robot.execute_single_cond(OUTER_COND):

                if self.inner_is_done:
                    reward = self.program.restart_outer(self.robot)
                    done = True
                else:
                    self.predicting_outer = False
                    self.predicting_inner = True

                    self.outer_is_done = True


            if not self.robot.execute_single_cond(OUTER_COND):
                
                # append and execute
                self.program.append_outer(action)
                self.robot.execute_single_action(action)

                if self.robot.execute_single_cond(OUTER_COND):
                    self.outer_is_done = True
                

        observation = self._get_obs()
        info = {}

        self.program.print()
        
        return observation, reward, done, info

    def _get_obs(self):
        robot_state = torch.stack([torch.from_numpy(self.robot.get_state())], dim=0)
        obs = self.encoder(robot_state.float()).detach().numpy()
        return obs

    def reset(self):
        self.seed += 1
        self.robot = KarelRobot(task=self.task, seed=self.seed)
        self.program = copy.deepcopy(self.init_program)

        self.inner_is_done = False
        self.outer_is_done = False

        self.predicting_inner = False
        self.predicting_outer = False

        return self._get_obs()
