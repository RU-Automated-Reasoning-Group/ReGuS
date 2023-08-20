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
        self.cond_outer = k_cond(negation=False, cond=k_cond_without_not('front_is_clear'))
        self.cond_inner = k_cond(negation=False, cond=k_cond_without_not('no_markers_present'))
        self.stmts_inner = []
        self.stmts_outer = []

    def append_inner(self, action):
        assert isinstance(action, k_action)
        self.stmts_inner.append(action)

    def append_outer(self, action):
        assert isinstance(action, k_action)
        self.stmts_outer.append(action)

    def restart_inner(self, robot):
        while robot.execute_single_cond(self.cond_inner) and not robot.no_fuel():
            for s in self.stmts_inner:
                if not robot.no_fuel():
                    r = robot.execute_single_action(s)
                    if r == 1:
                        return True
    
        return False

    def restart_outer(self, robot):
        r = 0.0
        while robot.execute_single_cond(self.cond_outer) and not robot.no_fuel():
            while robot.execute_single_cond(self.cond_inner) and not robot.no_fuel():
                for s in self.stmts_inner:
                    r = robot.execute_single_action(s)
            for s in self.stmts_outer:
                r = robot.execute_single_action(s)

        return r

    def print(self, prefix=''):

        p_str = prefix + 'DEF run (m WHILE (front_is_clear) (w '
        
        p_str += 'WHILE (no_markers_present) (i '
        for s in self.stmts_inner:
            p_str += str(s)
            p_str += ' '
        p_str += 'i) '

        for s in self.stmts_outer:
            p_str += str(s)
            p_str += ' '
        p_str += 'w)'
        p_str += ' m)'
        
        print(p_str)
