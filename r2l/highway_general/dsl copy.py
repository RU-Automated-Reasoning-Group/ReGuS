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

    def __call__(self, env):
        # debug
        cur_vel = env.vehicle.speed
        if cur_vel > 25:
            cond_idx = 2
        elif cur_vel > 20:
            cond_idx = 1
        else:
            cond_idx = 0

        state = env.observation_type.observe()
        if self.cond == 'front_is_clear':
            # return np.sum(state[-1, 1, :3]) == 0
            # return np.sum(state[-1, 1, :4]) == 0
            return np.sum(state[cond_idx:, 1, :4]) == 0
        elif self.cond == 'left_is_clear':
            # return np.sum(state[-1, 0, :3]) == 0
            # return np.sum(state[-1, 0, :4]) == 0
            return np.sum(state[cond_idx:, 0, :4]) == 0
        elif self.cond == 'right_is_clear':
            # return np.sum(state[-1, 2, :3]) == 0
            # return np.sum(state[-1, 2, :4]) == 0
            return np.sum(state[cond_idx:, 2, :4]) == 0
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

    def __call__(self, env, robot=None):
        if self.action == 'null':
            return None, robot.check_reward()

        if not robot.no_fuel():
            state, reward, done, _, info = env.step(self.action)

            # debug
            # plt.figure()
            # plt.imshow(env.render())
            # plt.savefig('store/direct_store/highway/{}_{}.png'.format(robot.action_steps, h_action.ACTION_LIST[self.action]))
            # print('{}_{}.png'.format(robot.action_steps, h_action.ACTION_LIST[self.action]))
            # plt.close()
            # if robot.action_steps > 50:
            # pdb.set_trace()

            if done:
                reward = -1
            if robot:
                # robot.steps += 1
                robot.action_steps += 1
            return state, reward
        else:
            return robot.get_state(), robot.check_reward()
                
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