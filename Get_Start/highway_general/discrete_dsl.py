import numpy as np
import matplotlib.pyplot as plt
import pdb

class h_cond:
    '''cond : cond_without_not
            | NOT C_LBRACE cond_without_not C_RBRACE
    '''
    def __init__(self, negation: bool, cond):
        self.negation = negation
        self.cond = cond

    def __call__(self, env):
        if self.negation:
            return not self.cond(env)
        else:
            return self.cond(env)

    def __str__(self):
        if self.negation:
            #return "NOT c( " + str(self.cond) + " c)"
            return "not (" + str(self.cond) + ")"
        else:
            return str(self.cond)

    def __eq__(self, other_cond):
        return self.negation == other_cond.negation and self.cond == other_cond.cond


class h_cond_without_not:
    '''cond_without_not : FRONT_IS_CLEAR
                        | LEFT_IS_CLEAR
                        | RIGHT_IS_CLEAR
    '''
    def __init__(self, cond: str):
        self.cond = cond

    def __call__(self, robot):
        # time_pos = np.ceil(robot.all_time_pos[robot.velocity]).astype(int)
        time_pos = 0
        if self.cond == 'front_is_clear':
            if np.sum(robot.state[robot.velocity, robot.lane_pos, time_pos+1: time_pos+4]) == 0:
                if robot.velocity == robot.state.shape[0] - 1:
                    return True
                elif np.sum(robot.state[robot.velocity+1, robot.lane_pos, time_pos+1: time_pos+4]) == 0:
                    return True
            return False

        elif self.cond == 'left_is_clear':
            if robot.lane_pos == robot.lane_domain[0]:
                return False
            if np.sum(robot.state[robot.velocity, robot.lane_pos-1, time_pos+1: time_pos+4]) == 0:
                if robot.velocity == robot.state.shape[0] - 1:
                    return True
                elif np.sum(robot.state[robot.velocity+1, robot.lane_pos-1, time_pos+1: time_pos+4]) == 0:
                    return True
            return False
        
        elif self.cond == 'right_is_clear':
            if robot.lane_pos == robot.lane_domain[-1]:
                return False
            if np.sum(robot.state[robot.velocity, robot.lane_pos+1, time_pos+1: time_pos+4]) == 0:
                if robot.velocity == robot.state.shape[0] - 1:
                    return True
                elif np.sum(robot.state[robot.velocity+1, robot.lane_pos+1, time_pos+1: time_pos+4]) == 0:
                    return True
            return False

        elif self.cond == 'all_true':
            return True

        else:
            raise NotImplementedError

    def __str__(self):
        return str(self.cond)


class h_action:
    '''action : LANE_LEFT
              | LANE_RIGHT
              | FASTER
              | SLOWER
              | IDLE
    '''
    ACTION_LIST = ['lane_left', 'idle', 'lane_right', 'faster', 'slower']

    def __init__(self, action: int):
        self.action = action

    def __call__(self, robot=None):
        if self.action == 'null':
            return None, robot.check_reward()

        if not robot.no_fuel():
            if self.action == 0:
                robot.lane_pos = max(robot.lane_pos-1, robot.lane_domain[0])
            elif self.action == 2:
                robot.lane_pos = min(robot.lane_pos+1, robot.lane_domain[-1])
            elif self.action == 3:
                robot.velocity = min(robot.velocity+1, robot.velocity_domain[-1])
            elif self.action == 4:
                robot.velocity = max(robot.velocity-1, robot.velocity_domain[0])

            # make a move on TTC
            robot.make_move()

            reward = robot.cal_cur_reward()
            robot.action_steps += 1

            return [robot.velocity, robot.lane_pos], reward
        
        else:
            [robot.velocity, robot.lane_pos], robot.cal_cur_reward()

                
    def __str__(self):
        # return ' ' + str(self.action)
        return ' ' + h_action.ACTION_LIST[self.action]


class h_if:
    '''if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE
    '''
    def __init__(self):
        #self.cond = None
        self.abs_state = []  # CNF, list of conds
        self.stmts = []

    def __call__(self, k, robot=None):
        if not robot.no_fuel:
            result = True
            for cond in self.abs_state:
                if not cond(k):
                    result = False
                    break
            if result:
                for s in self.stmts:
                    s(k, robot)

    # def __str__(self):
    #     return "IF c( " + str(self.cond) + " c) i( " + str(self.stmts) + " i)"

    def __str__(self):
        return 'TODO'