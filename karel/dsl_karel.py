import copy
import time
import math

from matplotlib.pyplot import isinteractive

from utils.logging import log_and_print
from karel.dsl import k_cond, k_cond_without_not, k_action

import pdb

ACTION_DICT = {
    'move'        : k_action('move'),
    'turn_right'  : k_action('turn_right'),
    'turn_left'   : k_action('turn_left'),
    'pick_marker' : k_action('pick_marker'),
    'put_marker'  : k_action('put_marker')
}
ACTION_NAME = [e for e in ACTION_DICT]
ACTION_LIST = [ACTION_DICT[e] for e in ACTION_DICT]

COND_DICT = {
    'front_is_clear'    : k_cond(negation=False, cond=k_cond_without_not('front_is_clear')),
    'left_is_clear'     : k_cond(negation=False, cond=k_cond_without_not('left_is_clear')),
    'right_is_clear'    : k_cond(negation=False, cond=k_cond_without_not('right_is_clear')),
    'markers_present': k_cond(negation=False, cond=k_cond_without_not('markers_present')),
    'not(front_is_clear)'    : k_cond(negation=True, cond=k_cond_without_not('front_is_clear')),
    'not(left_is_clear)'     : k_cond(negation=True, cond=k_cond_without_not('left_is_clear')),
    'not(right_is_clear)'    : k_cond(negation=True, cond=k_cond_without_not('right_is_clear')),
    'not(markers_present)': k_cond(negation=True, cond=k_cond_without_not('markers_present')),
    'all_true'          : k_cond(negation=False, cond=k_cond_without_not('all_true')),
}
COND_NAME = [e for e in COND_DICT]
COND_LIST = [COND_DICT[e] for e in COND_DICT]


class ABS_STATE:
    def __init__(self):
        self.state = {
            'front_is_clear'    : None,
            'left_is_clear'     : None,
            'right_is_clear'    : None,
            'markers_present'   : None,
        }

    def update(self, cond, description: str):
        # description: T / F / DNC
        self.state[str(cond)] = description

def get_abs_state(robot):
    abs_state = ABS_STATE()
    for cond in COND_LIST[:4]:
        if robot.execute_single_cond(cond):
            abs_state.update(cond, 'T')
        else:
            abs_state.update(cond, 'F')
    
    return abs_state


def satisfy_abs_state(current, required):
    satisfied = True
    for e in required.state:
        if required.state[e] == 'DNC':  # does not care
            pass
        elif current.state[e] != required.state[e]:  
            satisfied = False
            break

    return satisfied


def get_diff_abs_state(code_abs_state, obs_abs_state):
    diff_abs_state = []
    for e in code_abs_state.state:
        if obs_abs_state.state[e] != code_abs_state.state[e]:
            if code_abs_state.state[e] == 'DNC':
                pass
            else:
                diff_abs_state.append(e)

    return diff_abs_state


def get_diff_conds(code_abs_state, obs_abs_state):
    diff_conds = []  # can be multiple conds
    for e in code_abs_state.state:
        if obs_abs_state.state[e] != code_abs_state.state[e]:
            if code_abs_state.state[e] == 'DNC':
                pass
            elif code_abs_state.state[e] == 'T':
                diff_conds.append(COND_DICT['not(' + e + ')'])
            elif code_abs_state.state[e] == 'F':
                diff_conds.append(COND_DICT[e])

    return diff_conds


def get_neg_cond(cond):
    
    cond_name = str(cond)
    # print(cond_name)
    if 'not' in cond_name:
        neg_cond_name = cond_name.replace('not', '').replace('(', '').replace(')', '')
        neg_cond_name = neg_cond_name.replace(' ', '')
        return COND_DICT[neg_cond_name]
    else:
        cond_name = cond_name.replace(' ', '')
        for key in COND_DICT:
            if 'not' in key and cond_name in key:
                return COND_DICT[key]


def merge_abs_state(abs_state, new_abs_state):
    s = copy.deepcopy(abs_state)
    for e in s.state:
        if s.state[e] != new_abs_state.state[e]:
            s.state[e] = 'DNC'

    return s


class ACTION:
    def __init__(self, action):
        self.abs_state = None
        self.action = action
        
        # NOTE: used for adding new IF branch
        self.break_point = False
        self.obs_abs_state = None
        self.post_abs_state = None
        self.resume_point = False

    def execute(self, robot, stop):
        # if stop:
        #     pdb.set_trace()

        if robot.active and not robot.no_fuel():
            if robot.force_execution:  # without considering abs_state
                assert not self.resume_point
                r = robot.execute_single_action(self.action)
                if r >= robot.cur_goal or r == -1:
                    robot.active = False

                # robot.draw(log_print=True)
                # log_and_print("")
                # pdb.set_trace()

            elif not self.resume_point:
                # init abstract state
                if self.abs_state is None:
                    self.abs_state = get_abs_state(robot)

                # check satisfy
                if satisfy_abs_state(get_abs_state(robot), self.abs_state):
                    r = robot.execute_single_action(self.action)
                    # modify post abstrate state here (TODO: whether good to put here?)
                    new_robot_state = get_abs_state(robot)

                    # init post abstract state
                    if self.post_abs_state is None:
                        self.post_abs_state = new_robot_state
                    # update post abstract state
                    elif not satisfy_abs_state(new_robot_state, self.post_abs_state):
                        self.post_abs_state = merge_abs_state(self.post_abs_state, new_robot_state)

                    # NOTE: terminate when success (or failed)
                    if r >= robot.cur_goal or r == -1:
                        robot.active = False

                # add break point
                else:
                    self.break_point = True
                    self.bp_time = time.time()
                    self.obs_abs_state = get_abs_state(robot)
                    robot.active = False


    def reset_resume(self):
        self.resume_point = False

    def __str__(self):
        return str(self.action)

    def pretty_print(self):
        pass


# store action used to end if branch
class HIDE_ACTION:
    def __init__(self, action):
        self.abs_state = None
        self.action = action
        
        self.break_point = False
        self.obs_abs_state = None
        self.post_abs_state = None

    def execute(self, robot, stop):
        pass

    def __str__(self):
        return ''

    def pretty_print(self):
        pass


WRAPPED_ACTION_LIST = [ACTION(e) for e in ACTION_LIST]


# search DSL
# S -> while B do S; S | C
# B -> conds

# NOTE: treate C as terminals
#       C does not contribute to the sketch


class C:
    def __init__(self, act_num=math.inf):
        self.stmts = []
        self.act_num = act_num
        self.touch = False

        self.resume_point = False

    #def execute(self, robot):
    #    raise NotImplementedError('Invalid code')
    def execute(self, robot, stop):
        if not self.resume_point:
            assert robot.active
            robot.active = False
            self.touch = True

    def reset_resume(self):
        self.resume_point = False

    def __str__(self):
        return ' C '

class B:
    def __init__(self):
        self.cond = None

    def execute(self, robot):
        raise NotImplementedError('Invalid code')

    def __str__(self):
        return ' B ' if self.cond is None else str(self.cond)


class S:
    def __init__(self):
        self.stmts = []
    
    def execute(self, robot):
        if robot.active and not robot.no_fuel():
            #print('[execute S, i.e., execute nothing]')
            pass

    def __str__(self):
        return ' S '


class SN:
    def __init__(self):
        self.stmts = []
    
    def execute(self, robot):
        # raise NotImplementedError('Invalid code')
        if robot.active and not robot.no_fuel():
            #print('[execute S, i.e., execute nothing]')
            pass

    def __str__(self):
        return 'SN'


# NOTE: used to check if program complete
class END:
    def __init__(self):
        self.visited = False

    def execute(self, robot, stop):
        if robot.active:
            self.visited = True

    def __str__(self):
        return "; END"


state_store = []

class WHILE:
    def __init__(self):
        self.cond = [B()]
        self.stmts = [S()]
        self.robot_move = False
        self.resume_point = False
        self.resume_last = False
        self.start_pos = None

    def execute(self, robot, stop):
        # global state_store

        # for all_true
        resume_exist = False
        if str(self.cond[0]) == 'all_true':
            cur_action_step = robot.action_steps

        if robot.active and not robot.no_fuel():
            # check robot position
            if not self.resume_point:
                self.start_pos = robot.checker.get_hero_pos(robot.get_state())
            # do while
            while robot.active and (self.resume_point or (not robot.no_fuel() and robot.execute_single_cond(self.cond[0]))):
                for s in self.stmts:
                    # NOTE: summarized as
                    if stop:
                        robot.draw()
                        pdb.set_trace()

                    s.execute(robot, stop)

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
                if self.resume_last:
                    self.reset_resume()
                elif self.resume_point:
                    break

                # break if do nothing
                if str(self.cond[0]) == 'all_true' and cur_action_step == robot.action_steps:
                    if not resume_exist:
                        break
                    else:
                        resume_exist = False

                # state_store.append(robot.karel.state)

                # debug test
                robot.steps += 1
            
            # check robot position
            end_pos = robot.checker.get_hero_pos(robot.get_state())
            # update
            if not self.resume_point:
                if self.start_pos != end_pos:
                    self.robot_move = True
                else:
                    self.robot_move = False


    def reset_resume(self):
        self.resume_point = False
        self.resume_last = False
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
        self.cond = [B() if cond is None else cond]
        self.stmts = [C()]
        self.resume_point = False

    def execute(self, robot, stop):
        if robot.active and not robot.no_fuel():
            # IF
            if self.resume_point or robot.execute_single_cond(self.cond[0]):
                for s in self.stmts:
                    if stop:
                        robot.draw()
                        pdb.set_trace()
                    # NOTE: summarized as
                    s.execute(robot, stop)

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
        self.cond = [B() if cond is None else cond]
        self.stmts = [C()]
        self.else_stmts = [C()]
        self.resume_point = False

    def execute(self, robot, stop):
        if robot.active and not robot.no_fuel():
            into_if = False
            # IF
            if self.resume_point or robot.execute_single_cond(self.cond[0]):
                if not self.resume_point:
                    into_if = True
                for s in self.stmts:
                    if stop:
                        robot.draw()
                        pdb.set_trace()
                    # NOTE: summarized as
                    s.execute(robot, stop)

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
                    if stop:
                        robot.draw()
                        pdb.set_trace()
                    # NOTE: summarized as
                    s.execute(robot, stop)

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
        self.stmts = [S(), END()]
    
    def execute(self, robot, stop=False):
        # for debug
        # global state_store
        # state_store = []

        for s in self.stmts:
            if stop:
                robot.draw()
                pdb.set_trace()

            s.execute(robot, stop)

            # pdb.set_trace()
            if not robot.active:
                break
            if robot.no_fuel():
                break

        # state_store.append(robot.karel.state)

        # return state_store

    def complete(self):
        assert isinstance(self.stmts[-1], END)
        return self.stmts[-1].visited

    def reset(self):
        self.stmts[-1].visited = False

    def reset_c_touch(self):
        c_stmts, c_idx = self.find_actions(c_touch=True)
        while c_stmts is not None:
            c_stmts[c_idx].touch = False
            c_stmts, c_idx = self.find_actions(c_touch=True)

    # NOTE: find S / B
    def find(self):
        stmts, idx = self._find(self.stmts)
        if stmts is None:
            return None, None, None
        else:
            code = stmts[idx]
            if isinstance(code, S):
                code_type = 'S'
            elif isinstance(code, B):
                code_type = 'B'
            elif isinstance(code, SN):
                code_type = 'SN'
            else:
                raise ValueError('Invalid code')
            return stmts, idx, code_type

    def _find(self, stmts):
        r_stmts, r_idx = None, None
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, SN)):
                return stmts, idx
            elif isinstance(code, (WHILE, IF, IFELSE)):
                r_stmts, r_idx = self._find(code.cond)
                if r_stmts is None:
                    r_stmts, r_idx = self._find(code.stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
                if isinstance(code, IFELSE):
                    r_stmts, r_idx = self._find(code.else_stmts)
                    if not r_stmts is None:
                        return r_stmts, r_idx
            elif isinstance(code, (ACTION, k_cond, C, END)):
                pass
            else:
                raise ValueError('Invalid code')
        
        return r_stmts, r_idx
    
    # NOTE: find cond that contains a C
    #       C presents in IF(cond) {C}, not just in WHILE
    def find_c_cond(self, c_touch=False):
        self.c_cond, self.c_cond_type = None, None
        self.found_c = False
        self.c_touch = c_touch
        self._find_c_cond(self.stmts)

        return self.c_cond, self.c_cond_type

    def _find_c_cond(self, stmts):
        for code in stmts:
            if isinstance(code, (WHILE, IF, IFELSE)):
                contains_c = False
                for s in code.stmts:
                    if isinstance(s, C):
                        if not self.c_touch or s.touch:
                            contains_c = True
                            break
                if not self.found_c and contains_c:
                    self.c_cond, self.c_cond_type = code.cond[0], 'w' if isinstance(code, WHILE) else 'i'
                    self.found_c = True
                    return
                self._find_c_cond(code.stmts)
                if isinstance(code, IFELSE):
                    self._find_c_cond(code.else_stmts)
            elif isinstance(code, C):
                if not self.c_touch or code.touch:
                    self.found_c = True

    # find code containing C
    def find_c_stmt(self, cond_type, c_touch=False):
        self.found_c = False
        self.found_stmt = None
        self.c_touch = c_touch
        self._find_c_stmt(self.stmts, cond_type)
        assert self.found_stmt is not None

        return self.found_stmt

    def _find_c_stmt(self, stmts, cond_type):
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                pass
            elif isinstance(code, (IF, IFELSE)):
                self._find_c_stmt(code.stmts, cond_type)
                if self.found_c and self.found_stmt is None and cond_type=='i':
                    self.found_stmt = code
                if self.found_stmt is not None:
                    return
                if isinstance(code, IFELSE):
                    self._find_c_stmt(code.else_stmts, cond_type)
                    if self.found_c and self.found_stmt is None and cond_type=='i':
                        self.found_stmt = code
                    if self.found_stmt is not None:
                        return
            elif isinstance(code, WHILE):
                self._find_c_stmt(code.stmts, cond_type)
                if self.found_c and self.found_stmt is None and cond_type=='w':
                    self.found_stmt = code
                if self.found_stmt is not None:
                    return
            elif isinstance(code, C):
                if not self.c_touch or code.touch:
                    self.found_c = True
                    return
            else:
                pdb.set_trace()
                raise ValueError('Invalide code')


    # NOTE: find C
    def find_actions(self, c_touch=False):
        self.c_touch = c_touch
        stmts, idx = self._find_actions(self.stmts)
        if stmts is None:
            return None, None
        else:
            code = stmts[idx]
            assert isinstance(code, C)
            return stmts, idx

    def _find_actions(self, stmts):
        r_stmts, r_idx = None, None
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                pass
            elif isinstance(code, (WHILE, IF, IFELSE)):
                r_stmts, r_idx = self._find_actions(code.stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
                if isinstance(code, IFELSE):
                    r_stmts, r_idx = self._find_actions(code.else_stmts)
                    if not r_stmts is None:
                        return r_stmts, r_idx
            elif isinstance(code, C):
                if not self.c_touch or code.touch:
                    return stmts, idx
            else:
                pdb.set_trace()
                raise ValueError('Invalide code')
        
        return r_stmts, r_idx

    def find_break_point(self):
        stmts, idx = self._find_break_point(self.stmts)
        if stmts is None:
            return None, None
        else:
            code = stmts[idx]
            assert isinstance(code, ACTION) and code.break_point
            return stmts, idx

    def _find_break_point(self, stmts):
        r_stmts, r_idx = None, None
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, C, HIDE_ACTION, k_cond, END)):
                pass
            elif isinstance(code, (WHILE, IF, IFELSE)):
                r_stmts, r_idx = self._find_break_point(code.stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
                if isinstance(code, IFELSE):
                    r_stmts, r_idx = self._find_break_point(code.else_stmts)
                    if not r_stmts is None:
                        return r_stmts, r_idx
            elif isinstance(code, ACTION):
                if code.break_point:
                    return stmts, idx
            else:
                pdb.set_trace()
                raise ValueError('Invalide code')
        
        return r_stmts, r_idx


    # NOTE: expand sketch
    def expand(self):
        p_list = []
        new_prog = copy.deepcopy(self)
        stmts, idx, code_type = new_prog.find()
        # test parent sketch
        self_loop = False
        for code in new_prog.stmts:
            if isinstance(code, WHILE):
                self_loop = True
                break

        # expand B
        if code_type == 'B':
            for cond in COND_LIST:
                stmts[idx] = copy.deepcopy(cond)
                p_list.append(copy.deepcopy(new_prog))

        # expand S
        elif code_type == 'S':
            # S -> C
            if self_loop:
                stmts[idx] = C()
                p_list.append(copy.deepcopy(new_prog))
            # S -> while
            stmts[idx] = WHILE()
            # stmts.insert(idx + 1, SN())
            stmts.insert(idx + 1, S())
            p_list.append(copy.deepcopy(new_prog))

        # expand SN
        elif code_type == 'SN':
            pdb.set_trace()
            # SN -> S
            stmts[idx] = S()
            p_list.append(copy.deepcopy(new_prog))
            # SN -> None
            stmts.pop(idx)
            p_list.append(copy.deepcopy(new_prog))

        else:
            pass
        
        return p_list

    # NOTE: expand C to actions
    def expand_actions(self, c_touch=False, while_drop=True, keep_touch=False):
        p_list = []
        action_list = []
        new_prog = copy.deepcopy(self)
        # debug (for now, c_touch should be True)
        assert c_touch

        stmts, idx = new_prog.find_actions(c_touch)
        found_C = stmts[idx]

        if not stmts is None:
            for action in WRAPPED_ACTION_LIST:
                stmts[idx] = copy.deepcopy(action)
                # add C
                if found_C.act_num-1 > 0:
                    stmts.insert(idx+1, C(act_num=found_C.act_num-1))
                    if keep_touch:
                        stmts[idx+1].touch = True
                p_list.append(copy.deepcopy(new_prog))
                action_list.append(copy.deepcopy(action))
                # drop C
                if found_C.act_num-1 > 0:
                    stmts.pop(idx + 1)
            # attempt to drop C when While;C
            if while_drop and idx>0 and isinstance(stmts[idx-1], WHILE):
                stmts.pop(idx)
                p_list.append(copy.deepcopy(new_prog))
                action_list.append(None)
                stmts.insert(idx, C(act_num=found_C.act_num))
        else:
            pass
        
        return p_list, action_list

    def __str__(self):
        string = ''
        for s in self.stmts:
            string += str(s)
        
        return string

    # count C amount
    def count_C(self):
        self.count = 0
        self._count_C(self.stmts)

        return self.count

    def _count_C(self, stmts):
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                pass
            elif isinstance(code, (WHILE, IF, IFELSE)):
                self._count_C(code.stmts)
                if isinstance(code, IFELSE):
                    self._count_C(code.else_stmts)
            elif isinstance(code, C):
                self.count += 1
            else:
                pdb.set_trace()
                raise ValueError('Invalide code')

    # set resume point (c touch or break point)
    def set_resume_points(self):
        self.found = False
        path = self._set_resume_point(self.stmts, [])
        if not self.found:
            # due to prob mode
            print('prob mode error happen')
            # print('no c touch or break point')
            # pdb.set_trace()
            # print('solve?')
        else:
            for code in path:
                if hasattr(code, 'resume_point'):
                    code.resume_point = True

    def _set_resume_point(self, stmts, path):
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                if not isinstance(code, END) and code.break_point:
                    # print('why break point first??')
                    # pdb.set_trace()
                    # print('solve?')
                    # self.found = True
                    # return path
                    # due to prob mode
                    return []

                path.append(code)
                continue

            elif isinstance(code, IF):
                path.append(code)
                path = self._set_resume_point(code.stmts, path)
                if self.found:
                    return path

            elif isinstance(code, IFELSE):
                path.append(code)
                path = self._set_resume_point(code.stmts, path)
                if self.found:
                    return path
                path = self._set_resume_point(code.else_stmts, path)
                if self.found:
                    return path

            elif isinstance(code, WHILE):
                path.append(code)
                path = self._set_resume_point(code.stmts, path)
                if self.found:
                    # special case
                    if isinstance(code.stmts[-1], C) and code.stmts[-1].touch:
                        code.resume_last = True
                    return path

            elif isinstance(code, C):
                if code.touch:
                    self.found = True
                    return path
                else:
                    path.append(code)

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

            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
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

            elif isinstance(code, C):
                continue

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

            if isinstance(code, (S, B, ACTION, HIDE_ACTION, k_cond, END)):
                continue

            elif isinstance(code, (IF, IFELSE)):
                self._reset_resume_points(code.stmts)
                if isinstance(code, IFELSE):
                    self._reset_resume_points(code.else_stmts)

            elif isinstance(code, WHILE):
                self._reset_resume_points(code.stmts)

            elif isinstance(code, C):
                continue

            else:
                pdb.set_trace()
                raise ValueError('Invalide code')
