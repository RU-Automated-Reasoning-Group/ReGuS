from .environment import mine_env

class TopAction:
    def __init__(self, action):
        self.action = action

    def __str__(self):
        return ' '+self.action

class TopCondition:
    def __init__(self, neg, cond):
        self.neg = neg
        self.cond = cond

    def __str__(self):
        if self.neg:
            return 'not('+self.cond+')'
        return self.cond

class MinecraftRobot:
    def __init__(self, seed=0):
        self.env = mine_env(seed)
        self.rng = self.env.rng
        self.valid_actions = ['move', 'turn_left', 'turn_right', 'use']
        self.valid_perception = ['front_is_clear', 'left_is_clear', 'right_is_clear', \
                                 'front_is_door', 'left_is_goal', 'right_is_goal']

        # set position
        self.position, self.yaw = self.env.robot_start()
        self.robot_sign = {0:'>', 90:'v', 180:'<', 270:'^'}
        self.last_reward = 0

        # set others
        self.steps = 0
        self.max_steps = 1000
        self.active = True
        self.force_execution = False

        # for dsl
        self.while_start_robot_pos = None
        self.while_moved = False

    def no_fuel(self):
        return not self.steps <= self.max_steps

    # get next position
    def get_next_pos(self, pos, yaw):
        if yaw == 0:
            next_pos = [pos[0], pos[1]+1]
        elif yaw == 90:
            next_pos = [pos[0]+1, pos[1]]
        elif yaw == 180:
            next_pos = [pos[0], pos[1]-1]
        else:
            assert yaw == 270
            next_pos = [pos[0]-1, pos[1]]
        
        return next_pos

    def check_reward(self):
        return self.last_reward

    def execute_single_action(self, action):
        # move
        if action == self.valid_actions[0]:
            next_pos = self.get_next_pos(self.position, self.yaw)
            # attempt to move
            if self.env.get_obj(self.position) == 3:
                next_pos = self.position
            elif self.env.get_obj(next_pos) in [0, 2, 3, 4, 5]:
                next_pos = next_pos
            else:
                next_pos = self.position
            # get reward
            reward = self.env.get_reward(next_pos)
            # set
            self.position = next_pos
        # turn left
        elif action == self.valid_actions[1]:
            self.yaw = (360 + (self.yaw - 90)) % 360
            reward = self.last_reward
        # turn right
        elif action == self.valid_actions[2]:
            self.yaw = (360 + (self.yaw + 90)) % 360
            reward = self.last_reward
        # use
        elif action == self.valid_actions[3]:
            # attempt to react door
            self.env.door_react(self.position, self.yaw)
            reward = self.last_reward

        # check reward
        if self.last_reward == -1 or self.last_reward == 1:
            self.active = False
        elif reward == 1 or reward == -1:
            self.active = False
            self.last_reward = reward
        else:
            self.last_reward = reward

        return reward

    def execute_single_cond(self, cond, neg=False):
        # get position
        if 'front' in cond:
            check_yaw = self.yaw
        elif 'left' in cond:
            check_yaw = (360 + (self.yaw - 90)) % 360
        else:
            assert 'right' in cond
            check_yaw = (360 + (self.yaw + 90)) % 360
        check_pos = self.get_next_pos(self.position, check_yaw)

        # perception
        result = None
        if 'is_clear' in cond:
            if self.env.get_obj(self.position) == 3 and \
                (('front' in cond and self.yaw == 0) or \
                ('right' in cond and self.yaw == 270) or \
                ('left' in cond and self.yaw == 90)):
                result = False
            elif not self.env.get_obj(check_pos) in [0, 3, 4, 5]:
                result = False
            else:
                result = True
        elif 'is_door' in cond:
            if self.env.get_obj(self.position) == 3 and self.yaw == 0:
                result = True
            else:
                result = False
        elif 'is_goal' in cond:
            if self.env.get_obj(check_pos) == 4:
                result = True
            else:
                result = False

        if neg:
            return not result
        else:
            return result


    def draw(self, do_print=True):
        return self.env.draw(self.position, self.robot_sign[self.yaw], do_print)

if __name__ == '__main__':
    import pdb
    robot = MinecraftRobot(seed=1100)
    while True:
        pdb.set_trace()
        robot.draw()