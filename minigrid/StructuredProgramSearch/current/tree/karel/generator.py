import argparse
import os
import pdb
import pickle
import random

import h5py
import numpy as np
import progressbar


class KarelStateGenerator(object):
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def print_state(self, state=None):

        #agent_direction = {0: '^', 1: '>', 2: 'v', 3: '<'}
        agent_direction = {0: '^', 1: 'v', 2: '<', 3: '>'}
        
        state_2d = np.chararray(state.shape[:2])
        state_2d[:] = '.'
        state_2d[state[:,:,4]] = '#'
        for i in range(6, 16):
            state_2d[state[:,:,i]] = str(i - 5)


        x, y, z = np.where(state[:, :, :4] > 0)
        state_2d[x[0], y[0]] = agent_direction[z[0]]

        state_2d = state_2d.decode()
        for i in range(state_2d.shape[0]):
            print("".join(state_2d[i]))

    # generate an initial env
    def generate_single_state(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}):
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[:, :, 4] = self.rng.rand(h, w) > 1 - wall_prob
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True
        # Karel initial location
        valid_loc = False
        while(not valid_loc):
            y = self.rng.randint(0, h)
            x = self.rng.randint(0, w)
            if not s[y, x, 4]:
                valid_loc = True
                s[y, x, self.rng.randint(0, 4)] = True
        # Marker: num of max marker == 1 for now
        s[:, :, 6] = (self.rng.rand(h, w) > 0.9) * (s[:, :, 4] == False) > 0
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 5:]) == h*w, np.sum(s[:, :, :5])
        marker_weight = np.reshape(np.array(range(11)), (1, 1, 11))
        return s, y, x, np.sum(s[:, :, 4]), np.sum(marker_weight*s[:, :, 5:])

    # generate an initial env for cleanHouse problem
    def generate_single_state_clean_house(self, h=14, w=22, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for cleanHouse problem
        Valid program for cleanHouse problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( markersPresent c) i( pickMarker i) IFELSE c( leftIsClear c) i( turnLeft i) ELSE e( move e) w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """

        random.seed(self.seed)

        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',   0, '-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-', '-', '-', '-',   0, '-', '-'],
            ['-', '-',   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-', '-',   0,   0,   0,   0,   0,   0,   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-', '-', '-',   0, '-',   0, '-', '-', '-',   0, '-',   0,   0, '-', '-', '-',   0, '-',   0, '-', '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-', '-',   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ]

        assert h == 14 and w == 22, 'karel maze environment should be 13 x 13, found {} x {}'.format(h, w)
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h - 1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w - 1, 4] = True

        # Karel initial location
        agent_pos = (1, 13)
        hardcoded_invalid_marker_locations = [(1, 13), (2, 12), (3, 10), (4, 11), (5, 11), (6, 10)]
        s[agent_pos[0], agent_pos[1], 1] = True


        for y1 in range(h):
            for x1 in range(w):
                s[y1, x1, 4] = world_map[y1][x1] == '-'
                s[y1, x1, 5] = True if world_map[y1][x1] != 'M' else False
                s[y1, x1, 6] = True if world_map[y1][x1] == 'M' else False

        expected_marker_positions = set()
        for y1 in range(h):
            for x1 in range(13):
                if s[y1, x1, 4]:
                    if y1 - 1 > 0 and not s[y1 -1, x1, 4]: expected_marker_positions.add((y1 - 1,x1))
                    if y1 + 1 < h - 1 and not s[y1 +1, x1, 4]: expected_marker_positions.add((y1 + 1,x1))
                    if x1 - 1 > 0 and not s[y1, x1 - 1, 4]: expected_marker_positions.add((y1,x1 - 1))
                    if x1 + 1 < 13 - 1 and not s[y1, x1 + 1, 4]: expected_marker_positions.add((y1,x1 + 1))

        # put 2 markers near start point for end condition
        s[agent_pos[0]+1, agent_pos[1]-1, 5] = False
        s[agent_pos[0]+1, agent_pos[1]-1, 7] = True

        # place 10 Markers
        expected_marker_positions = list(expected_marker_positions)
        random.shuffle(expected_marker_positions)
        assert len(expected_marker_positions) >= 10
        marker_positions = []
        for i, mpos in enumerate(expected_marker_positions):
            if mpos in hardcoded_invalid_marker_locations:
                continue
            s[mpos[0], mpos[1], 5] = False
            s[mpos[0], mpos[1], 6] = True
            marker_positions.append(mpos)
            if len(marker_positions) == 10:
                break

        assert np.sum(s[:, :, 8:]) == 0
        metadata = {'agent_valid_positions': None, 'expected_marker_positions': expected_marker_positions, 'marker_positions': marker_positions}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env for fourCorners problem
    def generate_single_state_harvester(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for harvester problem
        Valid program for harvester problem:
        DEF run m( WHILE c( markersPresent c) w( WHILE c( markersPresent c) w( pickMarker move w) turnRight move turnLeft WHILE c( markersPresent c) w( pickMarker move w) turnLeft move turnRight w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        mode = env_task_metadata.get("mode", "train")
        marker_prob = env_task_metadata.get("train_marker_prob", 1.0) if mode == 'train' else env_task_metadata.get("test_marker_prob", 1.0)


        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        # initial karel position: karel facing east at the last row in environment
        agent_pos = (h-2, 1)
        s[agent_pos[0], agent_pos[1], 3] = True

        # put 1 marker at every location in grid
        if marker_prob == 1.0:
            s[1:h-1, 1:w-1, 6] = True
        else:
            valid_marker_pos = np.array([(r,c) for r in range(1,h-1) for c in range(1,w-1)])
            marker_pos = valid_marker_pos[np.random.choice(len(valid_marker_pos), size=int(marker_prob*len(valid_marker_pos)), replace=False)]
            for pos in marker_pos:
                s[pos[0], pos[1], 6] = True

        metadata = {}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env for randomMaze problem
    def generate_single_state_random_maze(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for random maze problem
        Valid program for random maze problem:
        DEF run m( WHILE c( noMarkersPresent c) w( IFELSE c( rightIsClear c) i( turnRight i) ELSE e( WHILE c( not c( frontIsClear c) c) w( turnLeft w) e) move w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """

        random.seed(self.seed)

        def get_neighbors(cur_pos, h, w):
            neighbor_list = []
            #neighbor top
            if cur_pos[0] - 2 > 0: neighbor_list.append([cur_pos[0] - 2, cur_pos[1]])
            # neighbor bottom
            if cur_pos[0] + 2 < h - 1: neighbor_list.append([cur_pos[0] + 2, cur_pos[1]])
            # neighbor left
            if cur_pos[1] - 2 > 0: neighbor_list.append([cur_pos[0], cur_pos[1] - 2])
            # neighbor right
            if cur_pos[1] + 2 < w - 1: neighbor_list.append([cur_pos[0], cur_pos[1] + 2])
            return neighbor_list

        s = np.zeros([h, w, 16]) > 0
        # convert every location to wall
        s[:, :, 4] = True
        #start from bottom left corner
        init_pos = [h - 2, 1]
        visited = np.zeros([h,w])
        stack = []

        # convert initial location to empty location from wall
        # iterative implementation of random maze generator at https://en.wikipedia.org/wiki/Maze_generation_algorithm
        s[init_pos[0], init_pos[1], 4] = False
        visited[init_pos[0], init_pos[1]] = True
        stack.append(init_pos)

        while len(stack) > 0:
            cur_pos = stack.pop()
            neighbor_list = get_neighbors(cur_pos, h, w)
            random.shuffle(neighbor_list)
            for neighbor in neighbor_list:
                if not visited[neighbor[0], neighbor[1]]:
                    stack.append(cur_pos)
                    s[(cur_pos[0]+neighbor[0])//2, (cur_pos[1]+neighbor[1])//2, 4] = False
                    s[neighbor[0], neighbor[1], 4] = False
                    visited[neighbor[0], neighbor[1]] = True
                    stack.append(neighbor)

        # init karel agent position
        agent_pos = init_pos
        s[agent_pos[0], agent_pos[1], 3] = True

        # Marker location
        valid_loc = False
        while (not valid_loc):
           ym = self.rng.randint(0, h)
           xm = self.rng.randint(0, w)
           if not s[ym, xm, 4]:
               valid_loc = True
               s[ym, xm, 6] = True
               assert not s[ym, xm, 4]

        assert not s[ym, xm, 4]

        # put 0 markers everywhere but 1 location
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 6]) == 1
        assert np.sum(s[:, :, 7:]) == 0
        metadata = {'agent_valid_positions': None}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env for fourCorners problem
    def generate_single_state_four_corners(self, h=12, w=12, wall_prob=0.1, env_task_metadata={}, is_top_off=False):
        """
        initial state generator for four corners problem
        Valid program for four corners problem:
        DEF run m( WHILE c( noMarkersPresent c) w( WHILE c( frontIsClear c) w( move w) IF c( noMarkersPresent c) i( putMarker turnLeft move i) w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        # initial karel position: karel facing east at the last row in environment
        agent_pos = (h-2, 2)
        s[agent_pos[0], agent_pos[1], 3] = True

        metadata = {}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env for stairClimer problem
    def generate_single_state_stair_climber(self, h=12, w=12, wall_prob=0.1, env_task_metadata={}):
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        random.seed(self.seed)

        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-', '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0, '-', '-',   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0, '-', '-',   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0, '-', '-',   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0, '-', '-',   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0, '-', '-',   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0, '-', '-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0, '-', '-',   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0, '-', '-',   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ]

        c = 2
        r = h - 1
        valid_agent_pos = []
        valid_init_pos = []
        while r > 0 and c < w:
            s[r, c, 4] = True
            s[r - 1, c, 4] = True
            if r - 1 > 0 and c - 1 > 0:
                valid_agent_pos.append((r - 1, c - 1))
                valid_init_pos.append((r - 1, c - 1))
                assert not s[r - 1, c - 1, 4] , "there shouldn't be a wall at {}, {}".format(r - 1, c - 1)
            if r - 2 > 0 and c - 1 > 0:
                valid_agent_pos.append((r - 2, c - 1))
                assert not s[r - 2, c - 1, 4], "there shouldn't be a wall at {}, {}".format(r - 2, c - 1)
            if r - 2 > 0 and c > 0:
                valid_agent_pos.append((r - 2, c))
                valid_init_pos.append((r - 2, c))
                assert not s[r - 2, c, 4], "there shouldn't be a wall at {}, {}".format(r - 2, c)
            c += 1
            r -= 1

        agent_valid_positions = list(set(valid_agent_pos))
        valid_init_pos = sorted(list(set(valid_init_pos)), key=lambda x: x[1])

        # Karel initial location
        l1, l2 = 0, 0
        while l1 == l2:
            l1, l2 = self.rng.randint(0, len(valid_init_pos)), self.rng.randint(0, len(valid_init_pos))
        agent_idx, marker_idx = min(l1, l2), max(l1, l2)
        agent_pos, marker_pos = valid_init_pos[agent_idx], valid_init_pos[marker_idx]
        assert (not s[agent_pos[0], agent_pos[1], 4]) and not (s[marker_pos[0], marker_pos[1], 4])
        s[agent_pos[0], agent_pos[1],  3] = True

        # Marker: num of max marker == 1 for now
        s[:, :, 5] = True
        s[marker_pos[0], marker_pos[1], 5] = False
        s[marker_pos[0], marker_pos[1], 6] = True

        assert np.sum(s[:, :, 6]) == 1
        assert np.sum(s[:, :, 7:]) == 0
        metadata = {'agent_valid_positions': agent_valid_positions}
        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env for TopOf problem
    def generate_single_state_top_off(self, h=12, w=12, wall_prob=0.1, env_task_metadata={}, is_top_off=True):
        """
        initial state generator for chain smoker and top off problem both
        Valid program for chain smoker problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( noMarkersPresent c) i( putMarker i) move w) m)
        Valid program for top off problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( MarkersPresent c) i( putMarker i) move w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        random.seed(self.seed)

        # initial karel position: karel facing east at the last row in environment
        agent_pos = (h-2, 1)
        s[agent_pos[0], agent_pos[1], 3] = True

        # randomly put markers at row h-2
        s[h-2, 1:w-1, 6] = self.rng.rand(w-2) > 1 - wall_prob
        # NOTE: need marker in last position as the condition is to check till I reach end
        s[h-2, w-2, 6] = True if not is_top_off else False
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:,:,5:]) == w*h

        # randomly generate wall at h-3 row
        mode = env_task_metadata.get('mode', 'train')
        hash_info_path = env_task_metadata.get('hash_info', None)
        if is_top_off and hash_info_path is not None:
            train_configs = env_task_metadata.get('train_configs', 1.0)
            test_configs = env_task_metadata.get('test_configs', 1.0)
            hash_info = pickle.load(open(hash_info_path,"rb"))
            assert hash_info['w'] == w and hash_info['h'] == h
            hashtable = hash_info['table']
            split_idx = int(len(hashtable)*train_configs) if mode == 'train' else int(len(hashtable)*test_configs)
            hashtable = hashtable[:split_idx] if mode == 'train' else hashtable[-split_idx:]
            key = s[h-2, 1:w-2, 6].tostring()
            if key not in hashtable:
                return self.generate_single_state_chain_smoker(h, w, wall_prob, env_task_metadata, is_top_off)

        # generate valid agent positions
        valid_agent_pos = [(h-2, c) for c in range(1, w-1)]
        agent_valid_positions = list(set(valid_agent_pos))
        # generate valid marker positions
        expected_marker_positions = [(h-2, c) for c in range(1, w-1) if not s[h-2, c, 6]]
        not_expected_marker_positions = [(h-2, c) for c in range(1, w-1) if s[h-2, c, 6]]
        metadata = {'agent_valid_positions':agent_valid_positions,
                    'expected_marker_positions':expected_marker_positions,
                    'not_expected_marker_positions': not_expected_marker_positions}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata

    # generate an initial env for TopOffPick problem
    def generate_single_state_top_off_pick(self, h=12, w=12, wall_prob=0.1, env_task_metadata={}, is_top_off=True):
        """
        initial state generator for chain smoker and top off problem both
        Valid program for chain smoker problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( noMarkersPresent c) i( putMarker i) move w) m)
        Valid program for top off problem:
        DEF run m( WHILE c( frontIsClear c) w( IF c( MarkersPresent c) i( putMarker i) move w) m)

        :param h:
        :param w:
        :param wall_prob:
        :return:
        """
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        random.seed(self.seed)

        # initial karel position: karel facing east at the last row in environment
        agent_pos = (h-2, 1)
        s[agent_pos[0], agent_pos[1], 3] = True

        # randomly put markers at row h-2
        marker_num = (self.rng.rand(w-2) > 1 - wall_prob).astype(int)
        for m_id in range(9):
            # pdb.set_trace()
            add_num = marker_num * (self.rng.rand(w-2) > 1 - wall_prob*(m_id+2)).astype(int) > 0
            if np.sum(add_num) == 0:
                break
            else:
                marker_num += add_num

        # pdb.set_trace()

        # add marker
        for c, m_n in enumerate(marker_num[:-1]):
            s[h-2, c+1, 5+m_n] = True


        # NOTE: need marker in last position as the condition is to check till I reach end
        s[h-2, w-2, marker_num[-1]] = True if not is_top_off else False
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:,:,5:]) == w*h

        # randomly generate wall at h-3 row
        mode = env_task_metadata.get('mode', 'train')
        hash_info_path = env_task_metadata.get('hash_info', None)
        if is_top_off and hash_info_path is not None:
            train_configs = env_task_metadata.get('train_configs', 1.0)
            test_configs = env_task_metadata.get('test_configs', 1.0)
            hash_info = pickle.load(open(hash_info_path,"rb"))
            assert hash_info['w'] == w and hash_info['h'] == h
            hashtable = hash_info['table']
            split_idx = int(len(hashtable)*train_configs) if mode == 'train' else int(len(hashtable)*test_configs)
            hashtable = hashtable[:split_idx] if mode == 'train' else hashtable[-split_idx:]
            key = s[h-2, 1:w-2, 6].tostring()
            if key not in hashtable:
                return self.generate_single_state_chain_smoker(h, w, wall_prob, env_task_metadata, is_top_off)

        # generate valid agent positions
        valid_agent_pos = [(h-2, c) for c in range(1, w-1)]
        agent_valid_positions = list(set(valid_agent_pos))
        # generate valid marker positions
        expected_marker_positions = [(h-2, c) for c in range(1, w-1) if not s[h-2, c, 6]]
        not_expected_marker_positions = [(h-2, c) for c in range(1, w-1) if s[h-2, c, 6]]
        metadata = {'agent_valid_positions':agent_valid_positions,
                    'expected_marker_positions':expected_marker_positions,
                    'not_expected_marker_positions': not_expected_marker_positions}

        return s, agent_pos[0], agent_pos[1], np.sum(s[:, :, 4]), metadata



if __name__ == "__main__":

    gen = KarelStateGenerator(seed=123)

    s, _, _, _, _ = gen.generate_single_state()
    #print(s.shape)
    #gen.print_state(s)
    
    s, _, _, _, _ = gen.generate_single_state_clean_house()
    #print(s.shape)
    #gen.print_state(s)
    
    s, _, _, _, _ = gen.generate_single_state_harvester()
    #print(s.shape)
    #gen.print_state(s)
    
    s, _, _, _, _ = gen.generate_single_state_random_maze()
    #print(s.shape)
    #gen.print_state(s)

    s, _, _, _, _ = gen.generate_single_state_four_corners()
    #print(s.shape)
    #gen.print_state(s)

    s, _, _, _, _ = gen.generate_single_state_top_off()
    #print(s.shape)
    #gen.print_state(s)

    s, _, _, _, _ = gen.generate_single_state_stair_climber()
    #gen.print_state(s)  # we need symbolic state to print
    #s = s.astype(int)  # convert to raw (digital) state input


