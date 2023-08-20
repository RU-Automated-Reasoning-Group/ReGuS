import copy

import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn

from karel.robot import KarelRobot
from karel.dsl import k_action, k_cond, k_cond_without_not
#from karel.program import AbsState, Node, Branch, Program
from karel.program import *


ACTION_INDEX = [0, 1, 2, 3, 4]
ACTION_NAME = [
    'move',
    'turn_right',
    'turn_left',
    'pick_marker',
    'put_marker'
]


COND_DICT = {
    'front_is_clear'    : k_cond(negation=False, cond=k_cond_without_not('front_is_clear')),
    'left_is_clear'     : k_cond(negation=False, cond=k_cond_without_not('left_is_clear')),
    'right_is_clear'    : k_cond(negation=False, cond=k_cond_without_not('right_is_clear')),
    'no_markers_present': k_cond(negation=False, cond=k_cond_without_not('no_markers_present')),
}


class KarelGymEnv(gym.Env):

    def __init__(self, task, program, seed, encoder):
        super().__init__()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float)

        self.task = task
        self.seed = seed
        self.encoder = encoder

        self.program = program
        self.init_program = copy.deepcopy(self.program)
        
        self.robot = KarelRobot(task=self.task, seed=self.seed)
        self.init_robot = copy.deepcopy(self.robot)  # to backup
        
        # TODO: not sure
        self.option = 'main'  # 'main' / 'build'

        # TODO: tmp branch creation and termination
        self.tmp_branch = None

        # TODO: try to count number of effective steps
        #self.effective_steps = 0
        
    def step(self, action, count_effective_steps):

        done = False
        reward = -0.01 #0.0

        # for quick access
        program = self.program
        robot = self.robot

        action = k_action(ACTION_NAME[action.item()])
        current_abs_state = self._get_abs_state(robot)

        #if count_effective_steps:
        #    self.early_return(0)
        
        # build the main branch
        if self.option == 'main':
            assert not program.main_branch_done
            
            # execute the newly generated code
            r = robot.execute_single_action(action)
            #if robot.moved:
            #    self.effective_steps += 1
            if r == 1:
                return self.early_return(1)

            post_abs_state = self._get_abs_state(robot)
            program.insert(current_abs_state, action, post_abs_state)  # TODO: always append
            
            # after finished the main branch, restart immediately, then
            # loop until detect we (may) need a new branch at some point
            # NOTE: restart here means start from scratch!

            # NOTE: stop when predicate is satisfied, and main branch is able to move
            if robot.execute_single_cond(program.cond): # and robot.moved:
                program.main_branch_done = True
                print('[MAIN]', len(program.main))
                program.print_main_branch()

                # TODO: optimize this
                # restart from scratch
                robot = copy.deepcopy(self.init_robot)
                r, break_point, info = program.execute(robot)
                #self.effective_steps += robot.effective_steps
                for key in info:
                    if key == 'finished' and info[key]:
                        return self.early_return(r)  # TODO: need to confirm this
                if r == 1:
                    return self.early_return(1)

                # don't forget to update break_point
                program.break_point = break_point
                
                self.option = 'build'
                
                observation = self._get_obs()
                info = {}

                return observation, reward, done, info

        # build a new branch
        if self.option == 'build':
            
            # execute the newly generated code
            abs_state = self._get_abs_state(robot)
            r = robot.execute_single_action(action)
            #if robot.moved:
            #    self.effective_steps += 1
            if r == 1:
                return self.early_return(1)
            post_abs_state = self._get_abs_state(robot)

            termination = action.action == program.break_point.action.action and \
                satisfy(post_abs_state, program.break_point.post_abs_state)
            
            if self.tmp_branch is None:
                # a new branch may be unnecessary
                if termination:
                    program.break_point.abs_state = abs_state_merge(program.break_point.abs_state, post_abs_state)

                    # TODO: optimize this, should be simplified
                    # restart from scratch
                    robot = copy.deepcopy(self.init_robot)
                    r, break_point, info = program.execute(robot)
                    #self.effective_steps += robot.effective_steps
                    for key in info:
                        if key == 'finished' and info[key]:
                            return self.early_return(r)  # TODO: need to confirm this
                    if r == 1:
                        #print('???')
                        return self.early_return(1)

                    # don't forget to update break_point
                    program.break_point = break_point
                    
                    self.option = 'build'
                    
                    # TODO: rethink the reward here
                    reward = r
                    observation = self._get_obs()
                    info = {}

                    return observation, reward, done, info

                # create/update a tmp branch
                else:
                    self.tmp_branch = Branch(abs_state)
            self.tmp_branch.insert(abs_state, action, post_abs_state)

            # terminate the new branch
            if termination:
                program.break_point.abs_state = abs_state_merge(program.break_point.abs_state, post_abs_state)
                program.break_point.branches.append(copy.deepcopy(self.tmp_branch))
                # TODO: under construction
                print('[SUB]', len(self.tmp_branch.nodes))
                for node in self.tmp_branch.nodes:
                    print(node.action)
                self.tmp_branch = None

                # TODO: optimize this
                # restart from scratch
                robot = copy.deepcopy(self.init_robot)
                r, break_point, info = program.execute(robot)
                #self.effective_steps += robot.effective_steps
                for key in info:
                    if key == 'finished' and info[key]:
                        return self.early_return(r)  # TODO: need to confirm this
                if r == 1:
                    #print('...')
                    return self.early_return(1)

                # don't forget to update break_point
                program.break_point = break_point

                self.option = 'build'

                observation = self._get_obs()
                info = {}

                return observation, reward, done, info

        observation = self._get_obs()
        info = {}

        return observation, reward, done, info

    def early_return(self, reward, meta_info=None):
        if meta_info:
            print(meta_info)
        #reward += self.effective_steps * 0.05
        return self._get_obs(), reward, True, {}

    def _get_obs(self):
        robot_state = torch.stack([torch.from_numpy(self.robot.get_state())], dim=0)
        obs = self.encoder(robot_state.float()).detach().numpy()
        return obs

    # TODO: under construction
    def _get_abs_state(self, robot):
        abs_state = AbsState()
        for cond in COND_DICT:
            # CNF
            if robot.execute_single_cond(COND_DICT[cond]):
                abs_state.update(cond, description=False)
            else:
                abs_state.update(cond, description=True)
        
        return abs_state

    def reset(self):
        self.seed += 1
        self.robot = KarelRobot(task=self.task, seed=self.seed)
        self.init_robot = copy.deepcopy(self.robot)  # to backup
        self.program = copy.deepcopy(self.init_program)
        self.option = 'main'
        self.tmp_branch = None
        #self.effective_steps = 0
        return self._get_obs()

