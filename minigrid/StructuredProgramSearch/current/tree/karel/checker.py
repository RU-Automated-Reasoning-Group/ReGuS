import pdb

import numpy as np
from scipy import spatial


class Checker:
    def __init__(self):
        pass

    @staticmethod
    def get_hero_pos(state):
        hero_pos = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                hero_info = state[i][j][:4]
                if hero_info.sum() == 1:
                    hero_pos = [i, j]
                    break

        return hero_pos

    @staticmethod
    def get_goal_pos(state):
        goal_pos = []
        for m in range(state.shape[0]):
            for n in range(state.shape[1]):
                if state[m][n][6] == 1:
                    goal_pos = [m, n]
                    break
        
        return goal_pos

    @staticmethod
    def get_markers_pos(state):
        markers_pos = []
        for m in range(1, state.shape[0] - 1):
            for n in range(1, state.shape[1] - 1):
                if state[m][n][6]:
                    markers_pos.append([m, n])
        
        return markers_pos


class RandomMazeChecker(Checker):
    def __init__(self, init_state):
        self.goal_pos = self.get_goal_pos(init_state)

    def __call__(self, state):
        reward = 1 if self.get_hero_pos(state) == self.goal_pos else 0
        reward = -1 if len(self.get_markers_pos(state)) > 1 else reward
        
        return reward

class AccRandomMazeChecker(Checker):
    def __init__(self, init_state):
        self.init_pos = self.get_hero_pos(init_state)
        self.goal_pos = self.get_goal_pos(init_state)

        self.min_distance_to_goal = spatial.distance.cityblock(self.init_pos, self.goal_pos)
        self.acc_r = 0

    def __call__(self, state):

        reward = 0
        current_pos = self.get_hero_pos(state)

        current_distance_to_goal = spatial.distance.cityblock(current_pos, self.goal_pos)
        if current_distance_to_goal == 0:
            reward = 1
            return reward

        if len(self.get_markers_pos(state)) > 1:
            reward = -1
            return reward
        
        # if current_distance_to_goal < self.min_distance_to_goal:
        #     self.min_distance_to_goal = current_distance_to_goal
        #     self.acc_r += 0.05

        # self.acc_r += 0.001  # not so sure

        return self.acc_r

class HarvesterChecker(Checker):
    def __init__(self, init_state):
        self.total_markers = np.sum(init_state[:,:,6:])

    def __call__(self, state):
        current_total_markers = np.sum(state[:,:,6:])
        reward = (self.total_markers - current_total_markers) / self.total_markers

        return reward


# NOTE: sparse reward, 1 means success, -1 means failed
class SparseStairClimberChecker(Checker):
    def __init__(self, init_state):
        self.goal_pos = self.get_goal_pos(init_state)
        self.valid_pos = []
        for m in range(1, init_state.shape[1] - 1):
            for n in range(1 , init_state.shape[0]):
                if init_state[m][n][4]:  # the last wall in a column
                    if n - 1 > 0:
                        self.valid_pos.append([m, n - 1])
                    if n - 2 > 0:
                        self.valid_pos.append([m, n - 2])
                    break
        
    def __call__(self, state):
        current_hero_pos = self.get_hero_pos(state)
        reward = 1 if current_hero_pos == self.goal_pos else 0
        reward = -1 if current_hero_pos not in self.valid_pos else reward

        return reward


# NOTE: dense reward
class StairClimberChecker(Checker):
    def __init__(self, init_state):
        self.init_state = init_state
        self.init_pos = self.get_hero_pos(init_state)
        self.marker_pos = self.get_goal_pos(init_state)
        
        self.valid_pos = []
        for m in range(1, init_state.shape[1] - 1):
            for n in range(1 , init_state.shape[0]):
                if init_state[m][n][4]:  # the last wall in a column
                    if n - 1 > 0:
                        self.valid_pos.append([m, n - 1])
                    if n - 2 > 0:
                        self.valid_pos.append([m, n - 2])
                    break
        
        self.min_distance_to_goal = spatial.distance.cityblock(self.init_pos, self.marker_pos)

    def __call__(self, state):

        reward = 0
        current_pos = self.get_hero_pos(state)

        if current_pos not in self.valid_pos:
            reward = -1
            return reward

        current_distance_to_goal = spatial.distance.cityblock(current_pos, self.marker_pos)
        
        if current_distance_to_goal < self.min_distance_to_goal:
            self.min_distance_to_goal = current_distance_to_goal
            reward = 0.5

        if current_distance_to_goal == 0:
            reward = 1

        return reward


# NOTE: dense reward, accumulated
class AccStairClimberChecker(Checker):
    def __init__(self, init_state):
        self.init_state = init_state
        self.init_pos = self.get_hero_pos(init_state)
        self.marker_pos = self.get_goal_pos(init_state)
        
        self.valid_pos = []
        for m in range(1, init_state.shape[1] - 1):
            for n in range(1 , init_state.shape[0]):
                if init_state[m][n][4]:  # the last wall in a column
                    if n - 1 > 0:
                        self.valid_pos.append([m, n - 1])
                    if n - 2 > 0:
                        self.valid_pos.append([m, n - 2])
                    break
        
        self.min_distance_to_goal = spatial.distance.cityblock(self.init_pos, self.marker_pos)
        self.acc_r = 0


    def __call__(self, state):

        reward = 0
        current_pos = self.get_hero_pos(state)

        if current_pos not in self.valid_pos:
            reward = -1
            return reward

        current_distance_to_goal = spatial.distance.cityblock(current_pos, self.marker_pos)
        if current_distance_to_goal == 0:
            reward = 1
            return reward
        
        if current_distance_to_goal < self.min_distance_to_goal:
            self.min_distance_to_goal = current_distance_to_goal
            self.acc_r += 0.05

        return self.acc_r


class FourCornerChecker(Checker):
    def __init__(self, init_state):
        pass

    def __call__(self, state):
        total_markers = np.sum(state[:,:,6:])
        reward = 0
        if state[1, 1, 6]:
            reward += 0.25
        if state[10, 1, 6]:
            reward += 0.25
        if state[10, 10, 6]:
            reward += 0.25
        if state[1, 10, 6]:
            reward += 0.25
        
        correct_markers = int(reward * 4)
        incorrect_markers = total_markers - correct_markers
        reward = 0 if incorrect_markers > 0 else reward
        
        return reward


class TopOffChecker(Checker):
    def __init__(self, init_state):
        self.markers_pos = self.get_markers_pos(init_state)
        self.num_markers = len(self.markers_pos)

    def __call__(self, state):
        reward = 0.0
        acc = True

        for m in range(1, state.shape[1] - 1):
            pos = [state.shape[0]- 2, m]
            if pos in self.markers_pos:
                if state[pos[0], pos[1], 7] and acc:
                    reward += 1
                else:
                    acc = False
            else:
                if not state[pos[0], pos[1], 5]:
                    acc = False

        if self.get_hero_pos(state) == [state.shape[0] -2, state.shape[1] -2]:
            if acc:
                reward += 1

        reward /= (self.num_markers + 1)

        return reward


class TopOffPickChecker(Checker):
    def __init__(self, init_state):
        self.markers_pos = self.get_markers_pos(init_state)
        self.total_markers = 0
        for m_id in range(10):
            self.total_markers += (m_id+1) * np.sum(init_state[:, :, m_id+6])
        # pdb.set_trace()

    def __call__(self, state):
        # get current markers
        current_total_markers = 0
        for m_id in range(10):
            current_total_markers += (m_id+1) * np.sum(state[:, :, m_id+6])

        # calculate reward
        if self.total_markers == 0:
            reward = 1 if current_total_markers == 0 else -1
        else:
            reward = max(-1, (self.total_markers - current_total_markers) / self.total_markers)
        # if reward > 0:
        #     pdb.set_trace()

        return reward


class CleanHouseChecker(Checker):
    def __init__(self, init_state):
        self.markers_pos = self.get_markers_pos(init_state)
        self.num_makers = len(self.markers_pos)

    def __call__(self, state):
        reward = 0

        # pdb.set_trace()

        for pos in self.markers_pos:
            if state[pos[0], pos[1], 5]:
                reward += 1

        reward /= len(self.markers_pos)
        
        return reward


class CleanHouseCheckerAdd(Checker):
    def __init__(self, init_state):
        self.markers_pos = self.get_markers_pos(init_state)
        self.num_makers = len(self.markers_pos)
        # get goal position
        self.goal_pos = self._get_self_goal(init_state)

    def _get_self_goal(self, state):
        goal_pos = []
        for m in range(state.shape[0]):
            for n in range(state.shape[1]):
                if state[m][n][7] == 1:
                    goal_pos = [m, n]
                    break
        
        return goal_pos

    def __call__(self, state):
        reward = 0

        for pos in self.markers_pos:
            if state[pos[0], pos[1], 5]:
                reward += 1

        reward /= len(self.markers_pos)
        
        # add goal position
        current_pos = self.get_hero_pos(state)
        reward = 0 if current_pos != self.goal_pos else reward

        return reward