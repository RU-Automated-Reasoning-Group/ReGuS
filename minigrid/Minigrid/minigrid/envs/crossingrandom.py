from __future__ import annotations

import itertools as itt

import numpy as np
import gymnasium

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv

import collections
import random
import torch


class CrossingRandomEnv(MiniGridEnv):

    """
    ## Description

    Depending on the `obstacle_type` parameter:
    - `Lava` - The agent has to reach the green goal square on the other corner
        of the room while avoiding rivers of deadly lava which terminate the
        episode in failure. Each lava stream runs across the room either
        horizontally or vertically, and has a single crossing point which can be
        safely used; Luckily, a path to the goal is guaranteed to exist. This
        environment is useful for studying safety and safe exploration.
    - otherwise - Similar to the `LavaCrossing` environment, the agent has to
        reach the green goal square on the other corner of the room, however
        lava is replaced by walls. This MDP is therefore much easier and maybe
        useful for quickly testing your algorithms.

    ## Mission Space
    Depending on the `obstacle_type` parameter:
    - `Lava` - "avoid the lava and get to the green goal square"
    - otherwise - "find the opening and get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of the map SxS.
    N: number of valid crossings across lava or walls from the starting position
    to the goal

    - `Lava` :
        - `MiniGrid-LavaCrossingS9N1-v0`
        - `MiniGrid-LavaCrossingS9N2-v0`
        - `MiniGrid-LavaCrossingS9N3-v0`
        - `MiniGrid-LavaCrossingS11N5-v0`

    - otherwise :
        - `MiniGrid-SimpleCrossingS9N1-v0`
        - `MiniGrid-SimpleCrossingS9N2-v0`
        - `MiniGrid-SimpleCrossingS9N3-v0`
        - `MiniGrid-SimpleCrossingS11N5-v0`

    """

    def __init__(
        self,
        size=9,
        num_crossings=1,
        obstacle_type=Lava,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type

        if obstacle_type == Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
            mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 100

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "find the opening and get to the green goal square"

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[: self.num_crossings]  # sample random rivers
        rivers_v = sorted(pos for direction, pos in rivers if direction is v)
        rivers_h = sorted(pos for direction, pos in rivers if direction is h)
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1])
                )
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1])
                )
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        # Make the current to be donoe
        self.grid.set(width - 2, height - 2, None)

        # Use BFS to find all locations that can be reached from the initial location
        # then, randomly sample a goal from these locations
        visited = set()

        q = collections.deque([self.agent_pos])
        available_goal = []
        while len(q) > 0:
            location = q.popleft()
            visited.add(tuple(location))
            # check all four directions
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dir in directions:
                new_x = location[0] + dir[0]
                new_y = location[1] + dir[1]
                if (new_x, new_y) not in visited:
                    if new_x >= 0 and new_x < width and new_y >= 0 and new_y < height:
                        if self.grid.get(new_x, new_y) is None:
                            # this is a square that can be reached by agent
                            new_location = np.array((new_x, new_y))
                            q.append(new_location)
                            if not np.array_equal(new_location, self.agent_pos):
                                available_goal.append(new_location)

        choice = self.np_random.choice(len(available_goal))
        goal_location = available_goal[choice]
        self.put_obj(Goal(), goal_location[0], goal_location[1])

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )


class CrossingRandomR2LEnv(CrossingRandomEnv):
    def __init__(
        self,
        size=9,
        num_crossings=1,
        obstacle_type=Lava,
        max_steps: int | None = None,
        **kwargs,
    ):

        super().__init__(
            size=size,
            num_crossings=num_crossings,
            obstacle_type=obstacle_type,
            max_steps=max_steps,
            **kwargs,
        )

        # set action space and observation space for r2l agent
        
        # turn_left, turn_right, move
        self.action_space = gymnasium.spaces.Discrete(3)

        # front_is_clear, left_is_clear, right_is_clear, goal_on_left, goal_on_right, goal_present
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(6, ))


    def step(self, action):
        # import pdb
        # pdb.set_trace()
        if action == 0:
            # turn_left
            real_action = self.actions.left
        elif action == 1:
            # turn_right
            real_action = self.actions.right
        elif action == 2:
            # move
            real_action = self.actions.forward
        else:
            assert False
        _, rwd, terminated, truncated, info = CrossingRandomEnv.step(self, real_action)
        return self.public_get_abs_obs(), rwd, terminated, truncated, info
    
    def public_get_abs_obs(self):
        b_front_is_clear = self.front_is_clear()
        b_left_is_clear = self.left_is_clear()
        b_right_is_clear = self.right_is_clear()
        b_goal_on_left = self.goal_on_left()
        b_goal_on_right = self.goal_on_right()
        b_goal_present = self.goal_present()

        obs = [
            float(b_front_is_clear),
            float(b_left_is_clear),
            float(b_right_is_clear),
            float(b_goal_on_left),
            float(b_goal_on_right),
            float(b_goal_present),
        ]

        return np.array(obs)

    def reset(self, *, seed: int | None = None, options=None):
        CrossingRandomEnv.reset(self, seed=seed, options=options)
        return self.public_get_abs_obs()

class LavaCrossingRandomR2LEnv(CrossingRandomEnv):
    def __init__(
        self,
        size=9,
        num_crossings=1,
        obstacle_type=Lava,
        max_steps: int | None = None,
        **kwargs,
    ):

        super().__init__(
            size=size,
            num_crossings=num_crossings,
            obstacle_type=obstacle_type,
            max_steps=max_steps,
            **kwargs,
        )

        # set action space and observation space for r2l agent
        
        # turn_left, turn_right, move, RC_get
        self.action_space = gymnasium.spaces.Discrete(3)

        # front_is_clear, left_is_clear, right_is_clear, goal_on_left, goal_on_right, goal_present, front_is_lava
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7, ))


    def step(self, action):
        # import pdb
        # pdb.set_trace()
        if action == 0:
            # turn_left
            real_action = self.actions.left
        elif action == 1:
            # turn_right
            real_action = self.actions.right
        elif action == 2:
            # move
            real_action = self.actions.forward
        elif action == 3:
            # RC_get
            policy_path = "MiniGrid-RandomCrossingR2LS11N5-v0/MiniGrid-RandomCrossingR2LS11N5-v0_seed_104_length_100.pt"
            normalizer_path = policy_path.replace(".pt", ".pkl")
            policy = torch.load(policy_path)
            policy.load_normalizer_param(path=normalizer_path)
            policy.init_hidden_state()
            # import pdb
            # pdb.set_trace()
            done = False
            state = torch.Tensor(self.public_get_abs_obs()[:-1])
            current_steps = 0
            while not done:
                action = policy(state, deterministic=True)
                next_state, reward, terminated, truncated, info = CrossingRandomEnv.step(self, action)
                done = terminated or truncated
                state = torch.Tensor(self.public_get_abs_obs()[:-1])
                current_steps += 1
            print(f"final rwd is {reward}")
            info["action_total_steps"] = current_steps
            return self.public_get_abs_obs(), reward, terminated, truncated, info
        else:
            assert False
        _, rwd, terminated, truncated, info = CrossingRandomEnv.step(self, real_action)
        return self.public_get_abs_obs(), rwd, terminated, truncated, info
    
    def public_get_abs_obs(self):
        b_front_is_clear = self.front_is_clear()
        b_left_is_clear = self.left_is_clear()
        b_right_is_clear = self.right_is_clear()
        b_goal_on_left = self.goal_on_left()
        b_goal_on_right = self.goal_on_right()
        b_goal_present = self.goal_present()
        b_front_is_lava = self.front_is_lava()

        obs = [
            float(b_front_is_clear),
            float(b_left_is_clear),
            float(b_right_is_clear),
            float(b_goal_on_left),
            float(b_goal_on_right),
            float(b_goal_present),
            float(b_front_is_lava),
        ]

        return np.array(obs)

    def reset(self, *, seed: int | None = None, options=None):
        CrossingRandomEnv.reset(self, seed=seed, options=options)
        return self.public_get_abs_obs()