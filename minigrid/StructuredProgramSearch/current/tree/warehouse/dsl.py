import copy

from karel.dsl import k_action, k_cond, k_cond_without_not
from karel.robot import KarelRobot

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
    'no_markers_present': k_cond(negation=False, cond=k_cond_without_not('no_markers_present')),
    'not(front_is_clear)'    : k_cond(negation=True, cond=k_cond_without_not('front_is_clear')),
    'not(left_is_clear)'     : k_cond(negation=True, cond=k_cond_without_not('left_is_clear')),
    'not(right_is_clear)'    : k_cond(negation=True, cond=k_cond_without_not('right_is_clear')),
    'not(no_markers_present)': k_cond(negation=True, cond=k_cond_without_not('no_markers_present')),
}
COND_NAME = [e for e in COND_DICT]
COND_LIST = [COND_DICT[e] for e in COND_DICT]


# search DSL
# S -> while B do S; S | C
# B -> conds

# NOTE: treate C as terminals
#       C does not contribute to the sketch


class C:
    def __init__(self):
        self.stmts = []

    def execute(self, robot):
        raise NotImplementedError('Invalid code')

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
        print('[execute S, i.e., execute nothing]')
        pass

    def __str__(self):
        return ' S '


class WHILE:
    def __init__(self):
        self.cond = [B()]
        self.stmts = [S()]

    def execute(self, robot):
        print('[execute WHILE]')
        while robot.execute_single_cond(self.cond[0]):
            for s in self.stmts:
                if isinstance(s, k_action):
                    robot.execute_single_action(s)
                else:
                    s.execute(robot)

    def __str__(self):
        string = ''
        string += ' WHILE(' + str(self.cond[0]) + ') {'
        for s in self.stmts:
            string += str(s)
        string += '} ;'

        return string


# NOTE: we will not synthesize IF directly
# TODO: consider this in find & expand functions
class IF:
    def __init__(self):
        self.cond = [B()]
        self.stmts = [S()]

    def execute(self, robot):
        print('[execute IF]')
        if robot.execute_single_cond(self.cond[0]):
            for s in self.stmts:
                if isinstance(s, k_action):
                    robot.execute_single_action(s)
                else:
                    s.execute(robot)

    def __str__(self):
        string = ''
        string += ' IF(' + str(self.cond[0]) + ') {'
        for s in self.stmts:
            string += str(s)
        string += '} '

        return string


class Program:
    def __init__(self):
        self.stmts = [S()]
    
    def execute(self, robot):
        for s in self.stmts:
            s.execute(robot)
    
    # TODO: find C
    def find_actions(self):
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
            if isinstance(code, (S, B)):
                pass
            elif isinstance(code, WHILE):
                r_stmts, r_idx = self._find_actions(code.stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
            if isinstance(code, (k_action, k_cond, C)):
                return stmts, idx
            else:
                raise ValueError('Invalid code')
        
        return r_stmts, r_idx

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
            # elif isinstance(code, C):
            #     code_type = 'C'
            else:
                raise ValueError('Invalid code')
            return stmts, idx, code_type

    def _find(self, stmts):
        r_stmts, r_idx = None, None
        for idx, code in enumerate(stmts):
            # NOTE: no need to find C here
            if isinstance(code, (S, B)):
                return stmts, idx
            elif isinstance(code, WHILE):
                r_stmts, r_idx = self._find(code.cond)
                if r_stmts is None:
                    r_stmts, r_idx = self._find(code.stmts)
                if not r_stmts is None:
                    return r_stmts, r_idx
            elif isinstance(code, (k_action, k_cond, C)):
                pass
            else:
                raise ValueError('Invalid code')
        
        return r_stmts, r_idx

    def __str__(self):
        string = ''
        for s in self.stmts:
            string += str(s)
        
        return string

    def expand(self):
        p_list = []
        stmts, idx, code_type = self.find()
        if code_type == 'B':
            for cond in COND_LIST:
                stmts[idx] = copy.deepcopy(cond)
                p_list.append(copy.deepcopy(self))
        elif code_type == 'S':
            stmts[idx] = C()
            p_list.append(copy.deepcopy(self))
            stmts[idx] = WHILE()
            stmts.insert(idx + 1, S())
            p_list.append(copy.deepcopy(self))
        else:
            pass
        
        return p_list

# TODO: incorporate this function into Program class
def expand(program, stmts, idx, code_type):
    p_list = []
    if code_type == 'B':
        for cond in COND_LIST:
            stmts[idx] = copy.deepcopy(cond)
            p_list.append(copy.deepcopy(program))
    elif code_type == 'S':
        stmts[idx] = C()
        p_list.append(copy.deepcopy(program))
        stmts[idx] = WHILE()
        stmts.insert(idx + 1, S())
        p_list.append(copy.deepcopy(program))
    # NOTE: no need to expand C
    # elif code_type == 'C':
    #     for action in ACTION_LIST:
    #         stmts[idx] = copy.deepcopy(action)
    #         stmts.insert(idx + 1, C())
    #         p_list.append(copy.deepcopy(program))
    #         stmts.pop(idx + 1)
    else:
        # expansion finished
        pass

    return p_list


def find_test():
    p = Program()

    # 1)
    p.stmts = [S()]
    stmts, idx, code_type = p.find()
    print(stmts[idx], code_type)
    
    # 2)
    p.stmts = [WHILE(), S()]
    stmts, idx, code_type = p.find()
    print(stmts[idx], code_type)

    # 3)
    w = WHILE()
    w.cond = [COND_LIST[0]]
    w.stmts = [S()]
    p.stmts = [
        w,
        S()
    ]
    stmts, idx, code_type = p.find()
    print(stmts[idx], code_type)

    # 4)
    w = WHILE()
    w.cond = [COND_LIST[0]]
    w.stmts = [C()]
    p.stmts = [
        w,
        S()
    ]
    stmts, idx, code_type = p.find()
    print(stmts[idx], code_type)

    # 5)
    w = WHILE()
    w.cond = [COND_LIST[0]]
    w.stmts = [ACTION_LIST[0]]
    p.stmts = [
        w,
        S()
    ]
    stmts, idx, code_type = p.find()
    print(stmts[idx], code_type)

    # 5)
    w1 = WHILE()
    w1.cond = [COND_LIST[0]]
    w2 = WHILE()
    w1.stmts = [w2, S()]
    p.stmts = [
        w1,
        S()
    ]
    stmts, idx, code_type = p.find()
    print(stmts[idx], code_type)

    # 6)
    p.stmts = []
    stmts, idx, code_type = p.find()
    print(stmts, idx, code_type)

    # 7)
    p.stmts = [C()]
    stmts, idx, code_type = p.find()
    print(stmts, idx, code_type)

    # 8)
    w1 = WHILE()
    w1.cond = [COND_LIST[0]]
    w2 = WHILE()
    w2.cond = [COND_LIST[1]]
    w2.stmts = [C()]
    w1.stmts = [w2, S()]
    p.stmts = [
        w1,
        S()
    ]
    stmts, idx, code_type = p.find()
    print(stmts[idx], code_type)


def expand_test():

    program = Program()
    
    # NOTE: before expansion, make sure to
    #       keep the original copy

    # 1)
    p = copy.deepcopy(program)
    p.stmts = [S()]
    stmts, idx, code_type = p.find()
    print(stmts[idx], code_type)

    p_list = expand(p, stmts, idx, code_type)
    print(p_list)
    print(p_list[0].stmts, p_list[1].stmts)

    #  2)
    p = copy.deepcopy(program)
    p.stmts = [B()]
    stmts, idx, code_type = p.find()
    print(stmts[idx], code_type)
    p_list = expand(p, stmts, idx, code_type)
    print(p_list)
    for _p in p_list:
        print(_p.stmts)

    # 3)
    # while_w1 (...) {
    #   while_w2 (...) {
    #       C
    #   }
    #   S
    # }
    # S
    p = copy.deepcopy(program)
    w1 = WHILE()
    w1.cond = [COND_LIST[0]]
    w2 = WHILE()
    w2.cond = [COND_LIST[1]]
    w2.stmts = [C()]
    w1.stmts = [w2, S()]
    p.stmts = [
        w1,
        S()
    ]
    stmts, idx, code_type = p.find()
    p_list = expand(p, stmts, idx, code_type)
    for _p in p_list:
        print(_p.stmts[0].stmts)


def execute_test():
    
    # 1) Harvester
    p = Program()
    # while (front_is_clear) {
    #   pick_marker
    #   move
    # }
    # S

    w = WHILE()
    w.cond = [COND_LIST[0]]  # front_is_clear
    w.stmts = [ACTION_LIST[3], ACTION_LIST[0]]  # pick_marker, move
    p.stmts = [
        w,
        S()
    ]

    robot = KarelRobot(task='harvester', seed=999)
    robot.draw()
    p.execute(robot)
    robot.draw()

    # 2) topOff
    p = Program()
    # while (front_is_clear) {
    #   if (markers_present) {
    #       put_marker
    #   }  
    #   move
    # }
    # S

    w = WHILE()
    w.cond = [COND_LIST[0]]  # front_is_clear
    i = IF()
    i.cond = [COND_LIST[7]]  # not(no_markers_present), i.e., markers_present
    i.stmts = [ACTION_LIST[4]]  # put_marker
    w.stmts = [i, ACTION_LIST[0]]  # if..., move
    p.stmts = [
        w,
        S()
    ]

    robot = KarelRobot(task='topOff', seed=999)
    robot.draw()
    p.execute(robot)
    robot.draw()

    # 3) TODO: terminate when incomplete
    p = Program()
    # while (front_is_clear) {
    #   C
    # }
    # S


def print_test():

    # 1)
    p = Program()
    w1 = WHILE()
    w1.cond = [COND_LIST[0]]
    w2 = WHILE()
    w2.cond = [COND_LIST[1]]
    w2.stmts = [C()]
    w1.stmts = [w2, S()]
    p.stmts = [
        w1,
        S()
    ]
    
    print(p)

    # 2)
    p = Program()
    w1 = WHILE()
    w1.cond = [COND_LIST[0]]
    w2 = WHILE()
    w1.stmts = [w2, S()]
    p.stmts = [
        w1,
        S()
    ]    


# TODO: check copy and save
def enumerate_test():
    program = Program()

    program.stmts = [S()]
    print('[start]', program)

    p = copy.deepcopy(program)
    p_list_0 = expand(p, *p.find())
    for _p in p_list_0:
        print(_p)

    p = copy.deepcopy(p_list_0[0])
    p_list_0_1 = expand(p, *p.find())
    assert len(p_list_0_1) == 0

    p = copy.deepcopy(p_list_0[1])
    p_list_0_2 = expand(p, *p.find())
    for _p in p_list_0_2:
        print(_p)


def find_C_test():
    
    p = Program()

    # 1)
    # while (front_is_clear) {
    #   C
    # }
    # S
    w = WHILE()
    w.cond = [COND_LIST[0]]
    w.stmts = [C()]
    p.stmts = [
        w,
        S()
    ]
    stmts, idx, code_type = p.find()
    print(stmts[idx], code_type)

    stmts, idx = p.find_actions()
    print(stmts, idx, stmts[idx])

    # 2) more examples
    p = Program()
    w = WHILE()
    w.cond = [COND_LIST[0]]
    w.stmts = [S()]
    p.stmts = [
        w,
        S()
    ]
    print('[original]', p)
    p_list = p.expand()
    for _p in p_list:
        print('[expand]', _p)
    print('[keep original]', p)



if __name__ == "__main__":

    #find_test()

    #expand_test()

    #execute_test()

    #print_test()

    #enumerate_test()

    find_C_test()

 # NOTE: find C and it's WHILE
    # NOTE: only can be used to find WHILE(B){C} structure
    def find_actions_while(self):
        stmts, idx, w = self._find_actions_while(self.stmts)
        if stmts is None:
            return None, None, None
        else:
            code = stmts[idx]
            assert isinstance(code, C)
            return stmts, idx, w

    def _find_actions_while(self, stmts):
        r_stmts, r_idx, w = None, None, None
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, IF, ACTION, k_cond)):
                pass
            elif isinstance(code, WHILE):
                for r_idx, s in enumerate(code.stmts):
                    if isinstance(s, C):
                        w = code
                        return code.stmts, r_idx, w
                    # TODO: not so sure, check this later
                    elif isinstance(s, ACTION):
                        pass
                    else:
                        r_stmts, r_idx, w = self._find_actions_while(code.stmts)
                        if not r_stmts is None:
                            return r_stmts, r_idx, w
            else:
                raise ValueError('Invalide code')
        
        return r_stmts, r_idx, w