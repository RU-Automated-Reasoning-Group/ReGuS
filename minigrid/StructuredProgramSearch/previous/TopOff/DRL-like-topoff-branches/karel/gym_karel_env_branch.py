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


COND_CODE = [
    k_cond(negation=False, cond=k_cond_without_not('front_is_clear')),
    k_cond(negation=False, cond=k_cond_without_not('left_is_clear')),
    k_cond(negation=False, cond=k_cond_without_not('right_is_clear')),
    k_cond(negation=False, cond=k_cond_without_not('markers_present')),
    k_cond(negation=False, cond=k_cond_without_not('no_markers_present')),

    k_cond(negation=True, cond=k_cond_without_not('front_is_clear')),
    k_cond(negation=True, cond=k_cond_without_not('left_is_clear')),
    k_cond(negation=True, cond=k_cond_without_not('right_is_clear')),
    k_cond(negation=True, cond=k_cond_without_not('markers_present')),
    k_cond(negation=True, cond=k_cond_without_not('no_markers_present')),
]
WHILE_COND = COND_CODE[0]  # 'front_is_clear' in TopOff task


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
        self.while_init_abs_state = True  # This is always True        
        #self.init_abs_state = self._get_abs_state()

        self.option = 'default_prediction'  # default_prediction / prediction

        # TODO: under construction
        self.new_branch = []  # tmp new branch
        self.new_branch_cond = None

    def step(self, action):

        done = False
        reward = 0.0
        current_abs_state = self._get_abs_state()

        action = k_action(ACTION_NAME[action.item()])

        if self.option == 'default_prediction':
            
            self.program.append_default(action)
            self.program.abs_states.append(current_abs_state)  # record <pre-abs-state>
            r = self.robot.execute_single_action(action)
            if r == 1:
                self.program.print(prefix='[success in default branch][prediction]')
                return self._get_obs(), 1, True, {}
                
            # check fuel
            if self.robot.no_fuel():
                self.program.print(prefix='[no fuel in default branch][prediction]')
                return self._get_obs(), 0, True, {}
                
            # NOTE: this is required
            recent_r = r

            # then try to restart carefully
            if self.robot.execute_single_cond(WHILE_COND) == self.while_init_abs_state:

                default_code = self.program.default()
                while self.robot.execute_single_cond(WHILE_COND) and not self.robot.no_fuel():
                        
                    for i in range(len(default_code)):
                            
                        # check the fuel
                        # if self.robot.no_fuel():
                        #     self.program.print(prefix='[no fule in default branch][restart]')
                        #     return self._get_obs(), 0, True, {}
                            
                        # compare <pre-abs-state>
                        if current_abs_state == self.program.abs_states[i]:
                            r = self.robot.execute_single_action(default_code[i])
                            recent_r = r
                            if r == 1:
                                self.program.print(prefix='[success in default branch][restart]')
                                return self._get_obs(), 1, True, {}
                            if self.robot.no_fuel():
                                self.program.print(prefix='[no fuel in default branch][restart]')
                                return self._get_obs(), r, True, {}

                        # prepare to create a new branch
                        else:
                            self.option = 'prediction'
                            self.break_point = i
                            # NOTE: we assume only one cond is selected right now
                            for j in range(len(COND_CODE)):
                                if current_abs_state[j] != self.program.abs_states[i][j]:
                                    self.new_branch_cond = current_abs_state[j]

                            return self._get_obs(), recent_r, False, {}

                # TODO: restart is simply finished
                done = True

        elif self.option == 'prediction':
            
            # finish the new branch
            
            if self.robot.no_fuel():
                self.program.print(prefix='[no fuel in new branch][prediction]')
                return self._get_obs(), 0, True, {}
            else:

                self.new_branch.append(action)

                self.program.append_default(action)
                self.program.abs_states.append(current_abs_state)  # record <pre-abs-state>
                r = self.robot.execute_single_action(action)
                if r == 1:
                    self.program.print(prefix='[success in default branch][prediction]')
                    return self._get_obs(), 1, True, {}
            
            




        observation = self._get_obs()
        info = {}

        return observation, reward, done, info

    def _get_obs(self):
        robot_state = torch.stack([torch.from_numpy(self.robot.get_state())], dim=0)
        obs = self.encoder(robot_state.float()).detach().numpy()
        return obs

    def _get_abs_state(self):
        abs_state_vector = []
        for cond in COND_CODE:
            abs_state_vector.append(self.robot.execute_single_cond(cond))

        return abs_state_vector

    def reset(self):
        self.seed += 1
        self.robot = KarelRobot(task=self.task, seed=self.seed)
        self.program = copy.deepcopy(self.init_program)

        # for TopOff problem
        self.while_init_abs_state = True  # This is always True
        self.if_init_abs_state = self._get_abs_state()

        self.while_is_done = False

        return self._get_obs()
