import numpy as np


# TODO: stateslogger early termination when tasks are done
#     : by introduction other functions?
class KarelStatesLogger:
    def __init__(self):
        self.states = []
        self.reward = 0

    def log(self, state):
        self.states.append(state)

    def get_history(self):
        return self.states

    # check goal positions, etc
    def init(self, state):
        raise NotImplementedError

    # check if need early termination (success)
    def check(self, state):
        raise NotImplementedError

    def hero_pos(self, state):
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                hero_info = state[i][j][:4]
                if hero_info.sum() == 1:
                    hero_pos = [i, j]                

        return hero_pos

    def compute_reward(self):
        return self.reward


class RandomMazeLogger(KarelStatesLogger):
    def __init__(self):
        super().__init__()
    
    # goal position
    def init(self, init_state):
        for m in range(init_state.shape[0]):
            for n in range(init_state.shape[1]):
                markers_info = init_state[m][n][6:]
                if markers_info.sum() == 1:
                    self.goal_pos = [m, n]

    def check(self, state):
        done = self.hero_pos(state) == self.goal_pos
        if done:
            self.reward = 1
        return done


class HarvesterLogger(KarelStatesLogger):
    def __init__(self):
        super().__init__()

    def init(self, init_state):
        self.total_markers = np.sum(init_state[:,:,6:])

    def check(self, state):
        done = False
        current_total_markers = np.sum(state[:,:,6:])
        if current_total_markers == 0:
            done = True
            self.reward = 1

        return done


class StairClimberLogger(KarelStatesLogger):
    def __init__(self):
        super().__init__()

    # goal position & valid positions
    def init(self, init_state):
        for m in range(init_state.shape[0]):
            for n in range(init_state.shape[1]):
                markers_info = init_state[m][n][6:]
                if markers_info.sum() == 1:
                    self.goal_pos = [m, n]
        
        self.valid_pos = []
        for m in range(1, init_state.shape[1] - 1):
            for n in range(1 , init_state.shape[0]):
                if init_state[m][n][4]:  # the last wall in a column
                    if n - 1 > 0:
                        self.valid_pos.append([m, n - 1])
                    if n - 2 > 0:
                        self.valid_pos.append([m, n - 2])
                    break

    # TODO: https://github.com/clvrai/leaps/blob/main/karel_env/karel.py#L185
    def check(self, state):
        done = False
        hero_pos = self.hero_pos(state)
        if hero_pos not in self.valid_pos:
            self.reward = -1
            done = True
        elif hero_pos == self.goal_pos:
            self.reward = 1
            done = True
        
        return done


class FourCornerLogger(KarelStatesLogger):
    def __init__(self):
        super().__init__()
    
    def init(self, init_state):
        pass

    def check(self, state):
        done = False
        reward = 0
        total_markers = np.sum(state[:,:,6:])

        if state[1, 1, 6]:
            reward += 0.25
        if state[6, 1, 6]:
            reward += 0.25
        if state[6, 6, 6]:
            reward += 0.25
        if state[1, 6, 6]:
            reward += 0.25

        correct_markers = int(reward * 4)
        incorrect_markers = total_markers - correct_markers
        if incorrect_markers > 0:
            reward = 0

        if reward == 1:
            done = True
            self.reward = 1
        
        return done


class TopOffLogger(KarelStatesLogger):
    def __init__(self):
        super().__init__()
    
    def init(self, init_state):
        self.markers_pos = []
        for m in range(1, init_state.shape[0] - 1):
            for n in range(1, init_state.shape[1] - 1):
                if init_state[m][n][6]:
                        self.markers_pos.append([m, n])
        self.num_markers = len(self.markers_pos)
        
    def check(self, state):
        done = False
        reward = 0

        for m in range(1, state.shape[1] - 1):
            pos = [state.shape[0] - 2, m]
            if pos in self.markers_pos:
                if state[pos[0]][pos[1]][7]:
                    reward += 1
                else:
                    break
            else:
                if not state[pos[0]][pos[1]][5]:
                    reward += 1
                else:
                    break
        
        if self.hero_pos(state) == [state.shape[0] - 2, state.shape[1] - 2]:
            reward += 1
        
        if reward == state.shape[1] - 1:
            self.reward = 1
            done = True

        return done


class CleanHouseLogger(KarelStatesLogger):
    def __init__(self):
        super().__init__()

    def init(self, init_state):
        self.markers_pos = []
        for m in range(1, init_state.shape[0] - 1):
            for n in range(1, init_state.shape[1] - 1):
                if init_state[m][n][6]:
                        self.markers_pos.append([m, n])
        self.num_markers = len(self.markers_pos)

    def check(self, state):
        done = False
        num_cleaned = 0
        for pos in self.markers_pos:
            if not state[pos[0]][pos[1]][6] and state[pos[0]][pos[1]][5]:
                num_cleaned += 1
        if num_cleaned == self.num_markers:
            done = True
            self.reward = 1

        return done
        
