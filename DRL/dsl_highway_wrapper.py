import copy
import time

from matplotlib.pyplot import isinteractive

from utils.logging import log_and_print

from highway_general.dsl import h_cond, h_cond_without_not, h_action

import pdb

ACTION_DICT = {
    'lane_left'   : h_action(0),
    'idle'        : h_action(1),
    'lane_right'  : h_action(2),
    'faster'      : h_action(3),
    'slower'      : h_action(4)
}
ACTION_NAME = [e for e in ACTION_DICT]
ACTION_LIST = [ACTION_DICT[e] for e in ACTION_DICT]

COND_DICT = {
    'front_is_clear'    : h_cond(negation=False, cond=h_cond_without_not('front_is_clear')),
    'left_is_clear'     : h_cond(negation=False, cond=h_cond_without_not('left_is_clear')),
    'right_is_clear'    : h_cond(negation=False, cond=h_cond_without_not('right_is_clear')),
    'all_true'          : h_cond(negation=False, cond=h_cond_without_not('all_true')),
    'not(front_is_clear)'    : h_cond(negation=True, cond=h_cond_without_not('front_is_clear')),
    'not(left_is_clear)'     : h_cond(negation=True, cond=h_cond_without_not('left_is_clear')),
    'not(right_is_clear)'    : h_cond(negation=True, cond=h_cond_without_not('right_is_clear')),
}
COND_NAME = [e for e in COND_DICT]
COND_LIST = [COND_DICT[e] for e in COND_DICT]


class COND:
    def __init__(self, cond):
        self.cond = cond
        self.resume_point = False
        self.touch = False

    def execute(self, robot):
        if str(self.cond) == 'all_true':
            return robot.execute_single_cond(self.cond)

        if not robot.do_perc:
            robot.active = False
            self.touch = True
        else:
            if self.resume_point:
                return True
            return robot.execute_single_cond(self.cond)

    def __str__(self):
        return str(self.cond)

    def reset_resume(self):
        self.touch = False
        self.resume_point = False


class ACTION:
    def __init__(self, action):
        self.action = action
        
        # NOTE: used for adding new IF branch
        self.resume_point = False

    def execute(self, robot):
        if robot.active and not robot.no_fuel():
            if not self.resume_point:
                r = robot.execute_single_action(self.action)
                if r == 1 or r == -1:
                    robot.active = False

    def reset_resume(self):
        self.resume_point = False

    def __str__(self):
        return str(self.action)

    def pretty_print(self):
        pass

WRAPPED_ACTION_LIST = [ACTION(e) for e in ACTION_LIST]

# NOTE: used to check if program complete
class END:
    def __init__(self):
        self.visited = False

    def execute(self, robot):
        if robot.active:
            self.visited = True

    def __str__(self):
        return "; END"


state_store = []

class WHILE:
    def __init__(self):
        self.cond = []
        self.stmts = []
        self.robot_move = True
        self.resume_point = False

    def execute(self, robot):
        # global state_store

        # for all_true
        resume_exist = False
        if str(self.cond[0]) == 'all_true':
            cur_action_step = robot.action_steps

        # attemp to touch perception label
        self.cond[0].execute(robot)

        if robot.active and not robot.no_fuel():
            # do while
            while (self.resume_point or (not robot.no_fuel() and self.cond[0].execute(robot))) and robot.active:
                for s in self.stmts:
                    s.execute(robot)

                    # resume point
                    if self.resume_point:
                        resume_exist = True
                        if hasattr(s, 'resume_point') and not s.resume_point:
                            self.reset_resume()

                    if not robot.active:
                        break
                    if robot.no_fuel():
                        break

                # resume point
                if self.resume_point:
                    break

                # break if do nothing
                if str(self.cond[0]) == 'all_true' and cur_action_step == robot.action_steps:
                    if not resume_exist:
                        break
                    else:
                        resume_exist = False

                if not robot.active:
                    break
                if robot.no_fuel():
                    break
                # attemp to touch perception label
                self.cond[0].execute(robot)


    def reset_resume(self):
        self.resume_point = False
        for c in self.cond:
            if hasattr(c, 'resume_point'):
                if c.resume_point:
                    c.reset_resume()
                else:
                    break
        for s in self.stmts:
            if hasattr(s, 'resume_point'):
                if s.resume_point:
                    s.reset_resume()
                else:
                    break


    def __str__(self):
        string = ''
        string += ' WHILE(' + str(self.cond[0]) + ') {'
        for s in self.stmts:
            string += str(s)
        string += '} ;'

        return string


# NOTE: we will not synthesize IF directly
class IF:
    def __init__(self, cond=None):
        self.cond = [cond]
        self.stmts = []
        self.resume_point = False

    def execute(self, robot):
        # attempt to touch perception
        self.cond[0].execute(robot)

        if robot.active and not robot.no_fuel():
            # IF
            if self.resume_point or self.cond[0].execute(robot):
                for s in self.stmts:
                    # NOTE: summarized as
                    s.execute(robot)

                    # resume point
                    if self.resume_point:
                        if hasattr(s, 'resume_point') and not s.resume_point:
                            self.reset_resume()

                    if not robot.active:
                        break
                    if robot.no_fuel():
                        break

    def reset_resume(self):
        self.resume_point = False
        for c in self.cond:
            if hasattr(c, 'resume_point'):
                if c.resume_point:
                    c.reset_resume()
                else:
                    break
        for s in self.stmts:
            if hasattr(s, 'resume_point'):
                if s.resume_point:
                    s.reset_resume()
                else:
                    # pdb.set_trace()
                    break


    def __str__(self):
        string = ''
        string += ' IF(' + str(self.cond[0]) + ') {'
        for s in self.stmts:
            string += str(s)
        string += '} '

        return string


# NOTE: we will not synthesize IF directly
class IFELSE:
    def __init__(self, cond=None):
        self.cond = [cond]
        self.stmts = []
        self.else_stmts = []
        self.resume_point = False

    def execute(self, robot):
        # attempt to touch perception
        self.cond[0].execute(robot)

        if robot.active and not robot.no_fuel():
            into_if = False
            # IF
            if self.resume_point or self.cond[0].execute(robot):
                if not self.resume_point:
                    into_if = True
                for s in self.stmts:
                    # NOTE: summarized as
                    s.execute(robot)

                    # resume point
                    if self.resume_point:
                        if hasattr(s, 'resume_point') and not s.resume_point:
                            into_if = True
                            self.reset_resume()

                    if not robot.active:
                        break
                    if robot.no_fuel():
                        break
            #ELSE
            if self.resume_point:
                assert not into_if
            if not into_if:
                for s in self.else_stmts:
                    # NOTE: summarized as
                    s.execute(robot)

                    # resume point
                    if self.resume_point:
                        if hasattr(s, 'resume_point') and not s.resume_point:
                            self.reset_resume()

                    if not robot.active:
                        break
                    if robot.no_fuel():
                        break


    def reset_resume(self):
        self.resume_point = False
        for c in self.cond:
            if hasattr(c, 'resume_point'):
                if c.resume_point:
                    c.reset_resume()
                else:
                    break
        for s in self.stmts:
            if hasattr(s, 'resume_point'):
                if s.resume_point:
                    s.reset_resume()
                else:
                    break
        for s in self.else_stmts:
            if hasattr(s, 'resume_point'):
                if s.resume_point:
                    s.reset_resume()
                else:
                    break


    def __str__(self):
        string = ''
        string += ' IF(' + str(self.cond[0]) + ') {'
        for s in self.stmts:
            string += str(s)
        string += '} '
        string += 'ELSE {'
        for s in self.else_stmts:
            string += str(s)
        string += '}'

        return string


class Program:
    def __init__(self):
        self.stmts = [END()]
    
    def execute(self, robot):
        # for debug
        # global state_store

        for s in self.stmts:
            s.execute(robot)

            # pdb.set_trace()
            if not robot.active:
                break
            if robot.no_fuel():
                break

    def complete(self):
        assert isinstance(self.stmts[-1], END)
        return self.stmts[-1].visited

    def reset(self):
        self.stmts[-1].visited = False

    def __str__(self):
        string = ''
        for s in self.stmts:
            string += str(s)
        
        return string

    # set resume point (c touch or break point)
    def set_resume_points(self):
        self.found = False
        path = self._set_resume_point(self.stmts, [])
        if not self.found:
            # due to prob mode
            if not self.stmts[-1].visited:
                print('prob mode error happen')
        else:
            for code in path:
                if hasattr(code, 'resume_point'):
                    code.resume_point = True

    def _set_resume_point(self, stmts, path):
        for idx, code in enumerate(stmts):
            if isinstance(code, (ACTION, END)):
                path.append(code)

            elif isinstance(code, COND):
                if code.touch:
                    self.found = True
                    code.touch = False
                    return path
                else:
                    path.append(code)

            elif isinstance(code, IF):
                path = self._set_resume_point(code.cond, path)
                if self.found:
                    return path
                path.append(code)
                path = self._set_resume_point(code.stmts, path)
                if self.found:
                    return path

            elif isinstance(code, IFELSE):
                path = self._set_resume_point(code.cond, path)
                if self.found:
                    return path
                path.append(code)
                path = self._set_resume_point(code.stmts, path)
                if self.found:
                    return path
                path = self._set_resume_point(code.else_stmts, path)
                if self.found:
                    return path

            elif isinstance(code, WHILE):
                path = self._set_resume_point(code.cond, path)
                if self.found:
                    return path
                path.append(code)
                path = self._set_resume_point(code.stmts, path)
                if self.found:
                    return path

            else:
                pdb.set_trace()
                raise ValueError('Invalide code')

        return path

    # check resume point
    def check_resume_points(self):
        self.found_resume = False
        self._check_resume_points(self.stmts)

        return self.found_resume

    def _check_resume_points(self, stmts):
        for idx, code in enumerate(stmts):
            if hasattr(code, 'resume_point') and code.resume_point:
                self.found_resume = True
                return

            if isinstance(code, (ACTION, COND, END)):
                continue

            elif isinstance(code, (IF, IFELSE)):
                self._check_resume_points(code.stmts)
                if self.found_resume:
                    return
                if isinstance(code, IFELSE):
                    self._check_resume_points(code.else_stmts)
                    if self.found_resume:
                        return

            elif isinstance(code, WHILE):
                self._check_resume_points(code.stmts)
                if self.found_resume:
                    return

            else:
                pdb.set_trace()
                raise ValueError('Invalide code')


    # reset resume point
    def reset_resume_points(self):
        self.found_resume = False
        self._reset_resume_points(self.stmts)

        return self.found_resume

    def _reset_resume_points(self, stmts):
        for idx, code in enumerate(stmts):
            if hasattr(code, 'resume_point') and code.resume_point:
                code.resume_point = False

            if isinstance(code, (ACTION, COND, END)):
                continue

            elif isinstance(code, (IF, IFELSE)):
                self._reset_resume_points(code.stmts)
                if isinstance(code, IFELSE):
                    self._reset_resume_points(code.else_stmts)

            elif isinstance(code, WHILE):
                self._reset_resume_points(code.stmts)

            else:
                pdb.set_trace()
                raise ValueError('Invalide code')
