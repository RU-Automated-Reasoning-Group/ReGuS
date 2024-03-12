import numpy as np

class mine_env:
    def __init__(self, seed=None):
        # init
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.obj = {0:'emp', 1:'wall', 2:'lava', 3:'cls_door', 4:'goal', 5:'op_door'}
        self.vis_obj = {0:' ', 1:'x', 2:'L', 3:'d', 4:'g', 5:'p'}
        self.obs_choice = [0, 2, 3]
        # init map
        self.map_generate()

    def map_init(self):
        self.map = np.zeros((9, 25))
        # add wall
        self.map[:, 0] = 1
        self.map[:, -1] = 1
        self.map[0, :] = 1
        self.map[-1, :] = 1
        # init others
        self.obstacles = []
        self.goal_pos = []

    def map_generate(self):
        # init map
        self.map_init()
        # generate obstacles
        obs_num = len(self.obs_choice)
        self.obstacles = [self.obs_choice[self.rng.randint(obs_num)] for _ in range(2)]
        self.goal_pos = [self.rng.randint(5)+2, 22]
        # put into map
        self.map[self.goal_pos[0], self.goal_pos[1]] = 4
        for obs_id, obs in enumerate(self.obstacles):
            if obs == 0:
                self.map[1:8, (obs_id+1)*8] = 1
                hole_pos = self.rng.randint(6)
                self.map[1+hole_pos, (obs_id+1)*8] = 0
                self.map[2+hole_pos, (obs_id+1)*8] = 0
            elif obs == 3:
                self.map[1:8, (obs_id+1)*8] = 1
                # door_pos = self.rng.randint(7)
                door_pos = 6
                self.map[1+door_pos, (obs_id+1)*8] = 3
            else:
                self.map[1:8, (obs_id+1)*8] = 2
                hole_pos = self.rng.randint(4)
                self.map[2+hole_pos, (obs_id+1)*8] = 0
                self.map[3+hole_pos, (obs_id+1)*8] = 0
    
    def get_obj(self, pos):
        return self.map[pos[0], pos[1]]

    def get_reward(self, pos):
        if self.map[pos[0], pos[1]] == 4:
            return 1
        elif self.map[pos[0], pos[1]] == 2:
            return -1
        # debug
        elif pos[1] > 8 and pos[1] < 16:
            return 0.2
        elif pos[1] > 16:
            return 0.4
        else:
            return 0

    def door_react(self, pos, yaw):
        if yaw == 0:
            if self.get_obj(pos) == 3:
                self.map[pos[0], pos[1]] = 5
            elif self.get_obj(pos) == 5:
                self.map[pos[0], pos[1]] = 3

    def robot_start(self):
        robot_pos = [self.rng.randint(7)+1, 1]
        # robot_pos = [2, 1]
        robot_yaw = 0
        return robot_pos, robot_yaw

    def draw(self, robot_pos, robot_sign='>', do_print=False):
        # draw map
        map_draw = ''
        for row_id in range(len(self.map)):
            for col_id in range(len(self.map[row_id])):
                if row_id == robot_pos[0] and col_id == robot_pos[1]:
                    map_draw += robot_sign
                else:
                    map_draw += self.vis_obj[self.map[row_id, col_id]]
            map_draw += '\n'
        
        if do_print:
            print(map_draw)
        return map_draw