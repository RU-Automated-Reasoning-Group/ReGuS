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


# GT program from scratch
program = Program()  # cond == 'front_is_clear'

action = k_action('move')
abs_state = AbsState()
abs_state.state = {
            'not(front_is_clear)'    : [False],
            'not(left_is_clear)'     : [False],
            'not(right_is_clear)'    : [True],
            'not(no_markers_present)': [False, True],  # NOTE: works for both
}
program.insert(abs_state=abs_state, action=action, post_abs_state=None)

program.break_point = program.main[0].branches
action = k_action('put_marker')
abs_state = AbsState()
abs_state.state = {
            'not(front_is_clear)'    : [False],
            'not(left_is_clear)'     : [False],
            'not(right_is_clear)'    : [True],
            'not(no_markers_present)': [True],
}
program.insert(abs_state=abs_state, action=action, post_abs_state=None)
GT_program = program


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
        
        self.robot = copy.deepcopy(KarelRobot(task=self.task, seed=self.seed))
        self.robot_info = {'task':self.task, 'seed':self.seed}

        # TODO: not sure
        self.option = 'main'  # 'main' / 'build'

        # TODO: tmp branch creation and termination
        self.tmp_branch = None
        
    def step(self, action, final_step):

        done = False
        reward = 0

        # TODO: verify this, again
        # for quick access
        program = self.program

        action = k_action(ACTION_NAME[action.item()])
        current_abs_state = self._get_abs_state(self.robot)

        if final_step:
            r = self.robot.execute_single_action(k_action('null'))
            return self.early_return(r)
        
        # build the main branch
        if self.option == 'main':
            
            # execute the newly generated code
            r = self.robot.execute_single_action(action)
            if r == 1:
                return self.early_return(1)

            post_abs_state = self._get_abs_state(self.robot)
            program.insert(current_abs_state, action, post_abs_state)  # TODO: always append
            
            # after finished the main branch, restart immediately, then
            # loop until detect we (may) need a new branch at some point
            # NOTE: restart here means start from scratch!

            if self.robot.execute_single_cond(program.cond):

                # TODO: try to execute gt program?
                #robot = copy.deepcopy(KarelRobot(**self.robot_info))
                #r, break_point, info = GT_program.execute(robot)
                #print(r, break_point, info)
                #assert r == 1

                # TODO: optimize this
                # restart from scratch
                #program.print_main_branch()
                self.robot = copy.deepcopy(KarelRobot(**self.robot_info))
                r, break_point, info = program.execute(self.robot)
                
                #if len(program.main) == 1 and str(program.main[0].action) == 'move':
                    #pass
                    # print(program.main[0].abs_state.state)
                    # print(program.main[0].post_abs_state.state)
                    # print(current_abs_state.state)
                    # print(post_abs_state.state)
                    #self.robot.gen.print_state(self.robot.get_state(printing=True))
                    #print('[in]', id(self.robot))

                for key in info:
                    if key == 'finished' and info[key]:
                        return self.early_return(r)  # TODO: need to confirm this
                if r == 1:
                    # TODO: disable this for now
                    #print('impossible')
                    #exit()
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
            abs_state = self._get_abs_state(self.robot)
            r = self.robot.execute_single_action(action)
            if r == 1:
                return self.early_return(1)
            post_abs_state = self._get_abs_state(self.robot)

            termination = action.action == program.break_point.action.action and \
                satisfy(post_abs_state, program.break_point.post_abs_state)
            
            if self.tmp_branch is None:
                # a new branch may be unnecessary
                if termination:
                    program.break_point.abs_state = abs_state_merge(program.break_point.abs_state, post_abs_state)

                    # TODO: optimize this, should be simplified
                    # restart from scratch
                    robot = copy.deepcopy(KarelRobot(**self.robot_info))
                    r, break_point, info = program.execute(robot)
                    for key in info:
                        if key == 'finished' and info[key]:
                            return self.early_return(r)  # TODO: need to confirm this
                    if r == 1:
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
                self.tmp_branch = None

                # TODO: optimize this
                # restart from scratch
                robot = copy.deepcopy(KarelRobot(**self.robot_info))
                r, break_point, info = program.execute(robot)
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

        observation = self._get_obs()
        info = {}

        return observation, reward, done, info

    def early_return(self, reward, meta_info=None):
        if meta_info:
            print(meta_info)
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
        self.robot = copy.deepcopy(KarelRobot(task=self.task, seed=self.seed))
        self.robot_info = {'task':self.task, 'seed':self.seed}
        self.program = Program()
        self.option = 'main'
        self.tmp_branch = None
        return self._get_obs()

