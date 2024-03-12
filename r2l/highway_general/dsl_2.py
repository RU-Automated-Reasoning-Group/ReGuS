import numpy as np
import matplotlib.pyplot as plt
import pdb

def TTC_view(cond, env):
    # debug
    cur_vel = env.vehicle.speed
    all_speeds = env.vehicle.target_speeds
    if cur_vel > all_speeds[1]:
        cond_idx = 2
    elif cur_vel > all_speeds[0]:
        cond_idx = 1
    else:
        cond_idx = 0

    state = env.observation_type.observe()
    if cond == 'front_is_clear':
        return np.sum(state[cond_idx:, 1, :4]) == 0
    elif cond == 'left_is_clear':
        return np.sum(state[cond_idx:, 0, :4]) == 0
    elif cond == 'right_is_clear':
        return np.sum(state[cond_idx:, 2, :4]) == 0
    elif cond == 'all_true':
        return True
    else:
        raise NotImplementedError


# currently only for front_is_clear
def Grid_view(cond, env):
    if cond == 'all_true':
        return True
    assert cond == 'front_is_clear'
    
    # ego car
    cur_vel = env.vehicle.speed
    fast_vel = [vel for vel in env.vehicle.target_speeds if vel >= cur_vel]
    if len(fast_vel) == 0:
        fast_vel = cur_vel
    else:
        fast_vel = min(fast_vel)
    
    slow_vel = [vel for vel in env.vehicle.target_speeds if vel <= cur_vel]
    if len(slow_vel) == 0:
        slow_vel = cur_vel
    else:
        slow_vel = max(slow_vel)

    # present car
    state = env.observation_type.observe()
    vehicles = np.where(state[0] == 1)

    # find speed of ego car
    ego_car_state = [0.0, 0.0, 0.0, 0.0] + env.vehicle.direction.tolist()

    # compare with other car
    crash = False
    for veh_1, veh_2 in zip(vehicles[0].tolist(), vehicles[1].tolist()):
        _, x, y, vx, vy, cosh, sinh = state[:, veh_1, veh_2]
        # ego car
        if vx == 0 and vy == 0:
            continue
        
        # test whether cross under faster and slower velocity
        for vel in [fast_vel, slow_vel]:
            del_x = vel*ego_car_state[4] - cur_vel*ego_car_state[4]
            del_y = vel*ego_car_state[5] - cur_vel*ego_car_state[5]

            tmp_x = x
            tmp_y = y
            for _ in range(3):
                if ((-del_x + vx) + tmp_x) * tmp_x <= 1e-6 and ((-del_y + vy) + tmp_y) * tmp_y <= 1e-6:
                    crash = True
                    print('crash happen')
                    break
                tmp_x += (-del_x + vx)
                tmp_y += (-del_y + vy)
            if crash:
                break

        if crash:
            break

    return not crash


# simple way for grid view
def Grid_view_simple(cond, env):
    if cond == 'all_true':
        return True
    assert cond == 'front_is_clear'

    # present car
    state = env.observation_type.observe()
    vehicles = np.where(state[0] == 1)

    # find speed of ego car
    # ego_grid = env.ego_grid
    ego_grid = [state.shape[1] // 2, state.shape[2] // 2]

    # compare with other car
    crash = False
    for veh_1, veh_2 in zip(vehicles[0].tolist(), vehicles[1].tolist()):
        _, x, y, vx, vy, cosh, sinh = state[:, veh_1, veh_2]
        # ego car
        if vx == 0 and vy == 0:
            continue
        
        # consider distance
        if not(np.abs(veh_1 - ego_grid[0]) <= 3 and np.abs(veh_2 - ego_grid[1]) <= 3):
            continue

        # consider whether getting close
        if x * vx <= 1e-6 and y * vy <= 1e-6:
            crash = True
            break

    return not crash


# currently only for front_is_clear
def Grid_view_2(cond, env):
    if cond == 'all_true':
        return True
    assert cond == 'front_is_clear'
    
    # ego car
    cur_vel = env.vehicle.speed
    fast_vel = [vel for vel in env.vehicle.target_speeds if vel >= cur_vel]
    if len(fast_vel) == 0:
        fast_vel = cur_vel
    else:
        fast_vel = min(fast_vel)
    
    slow_vel = [vel for vel in env.vehicle.target_speeds if vel <= cur_vel]
    if len(slow_vel) == 0:
        slow_vel = cur_vel
    else:
        slow_vel = max(slow_vel)

    # present car
    state = env.observation_type.observe()
    vehicles = np.where(state[0] == 1)

    # find speed of ego car
    ego_car_state = [0.0, 0.0, 0.0, 0.0] + env.vehicle.direction.tolist()

    # compare with other car
    crash = False
    for veh_1, veh_2 in zip(vehicles[0].tolist(), vehicles[1].tolist()):
        _, x, y, vx, vy, cosh, sinh = state[:, veh_1, veh_2]
        # ego car
        if vx == 0 and vy == 0:
            continue
        
        # test whether cross under faster and slower velocity
        for vel in [fast_vel, slow_vel]:
            del_x = vel*ego_car_state[4] - cur_vel*ego_car_state[4]
            del_y = vel*ego_car_state[5] - cur_vel*ego_car_state[5]
            exact_vx = vx - del_x
            exact_vy = vy - del_y

            tmp_x = x
            tmp_y = y

            for _ in range(4):
                # consider distance
                dis_check = True
                if np.abs(ego_car_state[4]) >= 0.99:
                    if not(np.abs(tmp_x) <= 1e-6 and np.abs(tmp_y) <= 0.09):
                        dis_check = False
                else:
                    if not(np.abs(tmp_x) <= 0.09 and np.abs(tmp_y) <= 0.09):
                        dis_check = False

                # consider whether getting close
                dir_check = True
                if not(tmp_x * exact_vx <= 1e-6 and tmp_y * exact_vy <= 1e-6):
                    dir_check = False

                # consider crash time
                if dis_check and dir_check:
                    if np.abs(tmp_x / exact_vx) <= 1 or np.abs(tmp_y / exact_vy) <= 1:
                        crash = True
                        break

                # next 1 second
                tmp_x += exact_vx
                tmp_y += exact_vy

            if crash:
                break
        if crash:
            break

    return not crash


class h_cond:
    '''cond : cond_without_not
            | NOT C_LBRACE cond_without_not C_RBRACE
    '''
    def __init__(self, negation: bool, cond):
        self.negation = negation
        self.cond = cond

    def __call__(self, env, view='TTC'):
        if self.negation:
            return not self.cond(env, view)
        else:
            return self.cond(env, view)

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

    def __call__(self, env, view='TTC'):
        if view == 'TTC':
            return TTC_view(self.cond, env)
        elif view == 'Grid':
            # return Grid_view(self.cond, env)
            return Grid_view_2(self.cond, env)
            # return Grid_view_simple(self.cond, env)
        else:
            raise NotImplementedError

    def __str__(self):
        return str(self.cond)


class h_action:
    '''action : 
              | SLOWER
              | IDLE
              | FASTER
    '''
    ACTION_LIST = ['slower', 'idle', 'faster']

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
            # # if robot.action_steps > 50:
            # pdb.set_trace()

            if done:
                if env.vehicle.crashed:
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