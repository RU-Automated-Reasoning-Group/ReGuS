import torch
import torch.nn as nn

from karel.dsl import *
from karel.robot import KarelRobot


ACTION_INDEX = [0, 1, 2, 3, 4]
ACTION_NAME = [
    'move',
    'turn_right',
    'turn_left',
    'pick_marker',
    'put_marker'
]


# make your life easier first
class SimpleProgram:
    def __init__(self):
        self.cond_while = k_cond(negation=False, cond=k_cond_without_not('front_is_clear'))
        
        # NOTE: there is a default branch
        #       that does not require cond to execute
        self.default = []
        
        # NOTE: nested list that record <pre-abs-states>
        #       of each line of the default branch
        self.abs_states = []
        
        # NOTE: extended branches and conds
        self.branches = []
        self.conds = []

    def append_default(self, action):
        assert isinstance(action, k_action)
        self.default.append(action)

    def restart_while(self, robot):
        
        r = 0.0
        
        while robot.execute_single_cond(self.cond_while) and not robot.no_fuel():
            
            if robot.execute_single_cond(self.cond_if) and not robot.no_fuel():
                for s in self.stmts_if:
                    if not robot.no_fuel():
                        r = robot.execute_single_action(s)
                        if r == 1:
                            return r

            for s in self.stmts_while:
                if not robot.no_fuel():
                    r = robot.execute_single_action(s)
                    if r == 1:
                        return r

        return r

    def print(self, prefix=''):

        p_str = prefix + 'DEF run (m WHILE (front_is_clear) (w '
        
        # also, print lazily
        if self.stmts_if:
            p_str += 'IF (markers_present) (i '
            for s in self.stmts_if:
                p_str += str(s)
                p_str += ' '
            p_str += 'i) '

        for s in self.stmts_while:
            p_str += str(s)
            p_str += ' '
        p_str += 'w)'
        p_str += ' m)'
        
        print(p_str)
