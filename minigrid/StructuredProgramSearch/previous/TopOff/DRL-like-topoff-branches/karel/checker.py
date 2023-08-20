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
        
        return reward


class HarvesterChecker(Checker):
    def __init__(self, init_state):
        self.total_markers = np.sum(init_state[:,:,6:])

    def __call__(self, state):
        current_total_markers = np.sum(state[:,:,6:])
        reward = (self.total_markers - current_total_markers) / self.total_markers

        return reward


class OldStairClimberChecker(Checker):
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
        
        self.steps = 0
        #self.previous_distance_to_goal = spatial.distance.cityblock(self.init_pos, self.marker_pos)
        self.min_distance_to_goal = spatial.distance.cityblock(self.init_pos, self.marker_pos)

    # def __call__(self, state):

    #     reward = 0
    #     current_pos = self.get_hero_pos(state)

    #     if current_pos not in self.valid_pos:
    #         reward = -1
        
    #     current_distance_to_goal = spatial.distance.cityblock(current_pos, self.marker_pos)
    #     if current_distance_to_goal == 0:
    #         reward = 2

    #     return reward

    def __call__(self, state):

        reward = 0
        #done = False
        current_pos = self.get_hero_pos(state)

        if current_pos not in self.valid_pos:
            reward = -1
            #done = True
            return reward #, done

        current_distance_to_goal = spatial.distance.cityblock(current_pos, self.marker_pos)
        
        # if current_distance_to_goal < self.previous_distance_to_goal:
        #     reward = 1
        
        if current_distance_to_goal < self.min_distance_to_goal:
            self.min_distance_to_goal = current_distance_to_goal
            reward = 1

        # NOTE: do we need this?
        #if current_distance_to_goal == 0:
            #print('reached')
            #reward = 2
            #done = True
        
        #self.previous_distance_to_goal = current_distance_to_goal  # TODO: previous min
        
        return reward#, done


# under construction
class LeapsStairClimberChecker(Checker):
    def __init__(self, init_state):
        self.init_state = init_state
        self.init_pos = self.get_hero_pos(init_state)
        self.marker_pos = self.get_goal_pos(init_state)
        self.steps = 0
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
        
        self.steps += 1

        reward = -1 * spatial.distance.cityblock(self.get_hero_pos(state), self.marker_pos)

        # NOTE: need to do this to avoid high negative reward for first action
        if self.steps == 1:
            x, y, z = np.where(self.init_state[:, :, :4] > 0)
            prev_pos = np.asarray([x[0], y[0], z[0]])
            self.prev_pos_reward = -1 * spatial.distance.cityblock(prev_pos[:2], self.marker_pos)

        abs_reward = reward
        reward = self.prev_pos_reward-1.0 if self.get_hero_pos(state) not in self.valid_pos else reward
        reward = reward - self.prev_pos_reward
        self.prev_pos_reward = abs_reward
        done = abs_reward == 0
        if done:
            reward = 10

        return reward, done


class FourCornerChecker(Checker):
    def __init__(self, init_state):
        pass

    def __call__(self, state):
        total_markers = np.sum(state[:,:,6:])
        reward = 0
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
        reward = 0 if incorrect_markers > 0 else reward
        
        return reward


class NewFourCornerChecker(Checker):
    def __init__(self, init_state):
        pass

    def __call__(self, state):
        total_markers = np.sum(state[:,:,6:])
        reward = 0
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
        reward = 0 if incorrect_markers > 0 else reward
        done = reward == 1

        return reward, done


class NewNewFourCornerChecker(Checker):
    def __init__(self, init_state):
        self.corners = [[1, 1], [6, 1], [6, 6], [1, 6]]
        self.finished_corners = []
        self.nearest_corner = [6, 1]
        self.nearest_distance = 1  # distance to nearest corner

    def find_nearest(self, state):
        
        self.finished_corners.append(self.nearest_corner)
        
        distance = np.inf
        corner = None
        for c in self.corners:
            if c not in self.finished_corners:
                d = spatial.distance.cityblock(self.get_hero_pos(state), c) 
                if d < distance:
                    distance = d
                    corner = c
                    break

        return corner, distance

    # under construction
    # a great mess
    def __call__(self, state):

        reward = 0
        done = False
        current_pos = self.get_hero_pos(state)

        total_markers = np.sum(state[:,:,6:])
        correct_markers = 0
        if state[1, 1, 6]:
            correct_markers += 1
        if state[6, 1, 6]:
            correct_markers += 1
        if state[6, 6, 6]:
            correct_markers += 1
        if state[1, 6, 6]:
            correct_markers += 1
        if total_markers > correct_markers:
            done = True
            return reward, done
        if total_markers == 4 and correct_markers == 4:
            print('finished')
            reward = 1
            done = True

        # if satisfy, give reward and reset nearest_corner and nearest_distance
        if state[self.nearest_corner[0], self.nearest_corner[1], 6] == 1:
            self.nearest_corner, self.nearest_distance = self.find_nearest(state)
            reward = 1
        else:         
            d = spatial.distance.cityblock(current_pos, self.nearest_corner)
            if d < self.nearest_distance:  # getting closer
                reward = 1
                self.nearest_distance = d

        return reward, done


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

        if self.get_goal_pos(state) == [state.shape[0] -2, state.shape[1] -2]:
            reward += 1
            #if acc:
            #    reward += 1

        reward /= (self.num_markers + 1)

        return reward


    # NOTE: this is wrong
    def __call__old(self, state):
        reward = 0
        acc = True

        for m in range(1, state.shape[1] - 1):
            pos = [state.shape[0] - 2, m]
            if pos in self.markers_pos:
                if state[pos[0], pos[1], 7]:
                    if acc:
                        reward += 1
                else:
                    #break
                    acc = False
            else:
                if not state[pos[0], pos[1], 5]:
                    if acc:
                        reward += 1
                else:
                    #break
                    acc = False

        if self.get_hero_pos(state) == [state.shape[0] - 2, state.shape[1] - 2]:
            reward += 1

        reward /= state.shape[1] - 1
        
        return reward


class CleanHouseChecker(Checker):
    def __init__(self, init_state):
        self.markers_pos = self.get_markers_pos(init_state)
        self.num_makers = len(self.markers_pos)

    def __call__(self, state):
        reward = 0

        for pos in self.markers_pos:
            if state[pos[0], pos[1], 5]:
                reward += 1

        reward /= len(self.markers_pos)
        
        return reward