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

        # NOTE: this only means the default branch is complete
        self.default_is_done = False

        # TODO: under construction
        self.working_new_branch = False
        self.new_branch_is_done = False
        self.new_branch = []  # tmp new branch

    # start with an simple empty while loop
    def step(self, action):

        done = False
        reward = 0.0
        current_abs_state = self._get_abs_state()

        # build the new line of code
        action = k_action(ACTION_NAME[action.item()])

        # working on default branch first
        if not self.default_is_done:

            # of course, check the fuel and execute
            if self.robot.no_fuel():
                self.program.print(prefix='[no fuel in default branch][prediction]')
                return self._get_obs(), 0, True, {}         
            else:
                # append and record
                self.program.append_default(action)
                self.program.abs_states.append(current_abs_state)
            
                r = self.robot.execute_single_action(action)
                if r == 1:
                    self.program.print(prefix='[success in default branch][prediction]')
                    return self._get_obs(), 1, True, {}

            if self.robot.execute_single_cond(WHILE_COND) == self.while_init_abs_state:
                self.default_is_done = True
                
                # then, try to restart the branch carefully until a <pre-states> 
                # is different from its in default branch
            
                default_code = self.program.default()
                for i in range(len(default_code)):

                    # of course, check the fuel
                    if self.robot.no_fuel():
                        self.program.print(prefix='[no fuel]')
                        return self._get_obs(), 0, True, {}

                    # compare <pre-states>
                    if current_abs_state == self.program.abs_states[i]:
                        
                        r = self.robot.execute_single_action(default_code[i])
                        if r == 1:
                            self.program.print(prefix='[success in default branch][restart]')
                            return self._get_obs(), 1, True, {}

                    else:

                        # found an inconsistent <pre-states>, prepare to create a new branch
                        self.working_new_branch = True
                        self.break_point = i  # location to insert new code

        elif self.working_new_branch:
            
            # preparing for the new branch
            if not self.new_branch_is_done:
            
                # of course, check the fuel and execute
                if self.robot.no_fuel():
                    self.program.print(prefix='[no fuel in creating new branch][prediction]')
                    return self._get_obs(), 0, True, {}         
                else:
                    # append and execute
                    self.new_branch.append(action)        
                    r = self.robot.execute_single_action(action)
                    if r == 1:
                        self.program.print(prefix='[success in default branch][prediction]')
                        return self._get_obs(), 1, True, {}

                    if current_abs_state == self.program.abs_states[self.break_point]:
                        self.new_branch_is_done = True



            # insert the predicted code and form a new branch
            new_code = default_code[:i] + [action] + default_code[i:]
            new_code_cond = current_abs_state
                        
            self.branches.append(new_code)
            self.conds.append(new_code_cond)



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
