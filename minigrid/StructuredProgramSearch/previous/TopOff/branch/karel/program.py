from .dsl import *


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


# NOTE: currently, use CNF
class AbsState:
    def __init__(self):
        self.state = {
            'not(front_is_clear)'    : [],
            'not(left_is_clear)'     : [],
            'not(right_is_clear)'    : [],
            'not(no_markers_present)': [],
        }

    def update(self, cond: str, description: bool):
        cond_str = 'not(' + cond + ')'
        self.state[cond_str].append(description)


def satisfy(s1: AbsState, s2: AbsState):
    s1_state = s1.state
    s2_state = s2.state
    assert [key for key in s1_state] == [key for key in s2_state]
    
    satisfied = True
    for key in [key for key in s2_state]:
        if s2_state[key] == [True, False] or s2_state[key] == [False, True]:
           satisfied = satisfied and True
        else:
            satisfied = satisfied and s1_state[key] == s2_state[key] 
    
    return satisfied


def satisfy_any(s1: AbsState, s2s: list):
    satisfied = None
    satisfied_abs_state = None
    for s2 in s2s:
        if satisfy(s1, s2):
            if not satisfied is None:
                satisfied = satisfied or True
            else:
                satisfied = True
            satisfied_abs_state = s2
        else:
            if not satisfied is None:
                satisfied = satisfied or False
            else:
                satisfied = False

    if satisfied is None:
        satisfied = False

    return satisfied, satisfied_abs_state


def abs_state_merge(s1: AbsState, s2: AbsState):
    s3 = AbsState()
    for key in s1.state:
        s1_descriptions = s1.state[key]
        s2_descriptions = s1.state[key]
        
        if len(s1_descriptions) == 2 or len(s2_descriptions) == 2 \
            or s1_descriptions != s2_descriptions:
            s3.state[key] = [True, False]
        else:
            s3.state[key] = s1_descriptions

    return s3

class Node:
    def __init__(self, abs_state=None, action=None, post_abs_state=None):
        self.abs_state = abs_state
        self.post_abs_state = post_abs_state  # we need this to terminate a if branch
        self.action = action
        self.branches = []  # attached branches, aim to go back to self.abs_state
                            # and then execute self.action

    def execute(self, robot):
        if not robot.no_fuel():
            current_abs_state = self._get_abs_state(robot)

            satisfied_any, satisfied_abs_state = \
                satisfy_any(current_abs_state, [branch.abs_state for branch in self.branches])
            
            if satisfied_any:
                #print('execute IF code')
                
                executed_branch = None
                for branch in self.branches:
                    if branch.abs_state == satisfied_abs_state:
                        executed_branch = branch  # TODO: not sure, check this later
                        r, break_point, info = branch.execute(robot)  # TODO: under construction
                        if r == 1:
                            return 1, None, {}
                        elif not break_point is None:
                            return r, break_point, info
                
                # don't forget to execute pivot
                current_abs_state = self._get_abs_state(robot)
                if satisfy(current_abs_state, self.abs_state):
                    #print('do not forget the pivot')
                    r = robot.execute_single_action(self.action)
                    #print(r)
                    return r, None, {}
                else:
                    # TODO: rethink this situation
                    # we may need a new branch
                    #print('we may need a new sequential branch, [1]')
                    return 0, self, {}  # TODO: inside a Branch, nested
            
            elif satisfy(current_abs_state, self.abs_state):
                r = robot.execute_single_action(self.action)
                #print('execute pivot code', r)
                return r, None, {}
            
            else:
                # TODO: we may need a new branch
                #print('we may need a new parallel branch, [2]')
                #return 0, self.branches, {'abs_state':self.abs_state, 'action':self.action, 'post_abs_state':self.post_abs_state}  # TODO: inside Node.branches
                return 0, self, {}  # TODO: inside Node.branches
    
        else:
            # NOTE: no fuel, may have partial reward
            r = robot.execute_single_action(k_action('null'))
            return r, None, {}

    def _get_abs_state(self, robot):
        abs_state = AbsState()
        for cond in COND_DICT:
            # CNF
            if robot.execute_single_cond(COND_DICT[cond]):
                abs_state.update(cond, description=False)
            else:
                abs_state.update(cond, description=True)
        
        return abs_state


class Branch:
    def __init__(self, abs_state=None):
        self.abs_state = abs_state
        self.nodes = []

    def execute(self, robot):
        for node in self.nodes:
            r, break_point, info = node.execute(robot)
            if r == 1:
                return 1, None, {}
            elif not break_point is None:
                return r, break_point, info
        return r, None, {}

    def insert(self, abs_state, action, post_abs_state):
        node = Node(abs_state=abs_state, action=action, post_abs_state=post_abs_state)
        self.nodes.append(node)

    # under construction
    def print(self):
        branch_str = '[BRANCH]'
        for node in self.nodes:
            branch_str += '[' + str(node.action) + ']'
        print(branch_str)


class Program:
    def __init__(self, cond=COND_DICT['front_is_clear']):
        self.cond = cond
        self.main = []
        
        self.break_point = self.main  # NOTE: where should I insert new code

    def execute(self, robot):
        r = 0
        while robot.execute_single_cond(self.cond) and not robot.no_fuel():
            for node in self.main:
                r, break_point, info = node.execute(robot)
                if r == 1 or not break_point is None:
                    return r, break_point, info
            
        # return the final results
        r = robot.execute_single_action(k_action('null'))
        return r, None, {'finished': True}

    # insert code to current break_point
    def insert(self, abs_state, action, post_abs_state):
        node = Node(abs_state=abs_state, action=action, post_abs_state=post_abs_state)
        self.break_point.append(node)

    def empty(self):
        return len(self.main) == 0

    # TODO: tmp code
    def print_main_branch(self):
        main_str = '[MAIN]'
        for node in self.main:
            main_str += '[' + str(node.action) + ']'
        print(main_str)
    