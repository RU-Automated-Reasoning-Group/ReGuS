from __future__ import annotations

import gymnasium
import numpy as np
import torch

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv


class DoorKeyEnv(MiniGridEnv):

    """
    ## Description

    This environment has a key that the agent must pick up in order to unlock a
    goal and then get to the green goal square. This environment is difficult,
    because of the sparse reward, to solve using classical RL algorithms. It is
    useful to experiment with curiosity or curriculum learning.

    ## Mission Space

    "use the key to open the door and then get to the goal"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

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
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-DoorKey-5x5-v0`
    - `MiniGrid-DoorKey-6x6-v0`
    - `MiniGrid-DoorKey-8x8-v0`
    - `MiniGrid-DoorKey-16x16-v0`

    """

    def __init__(self, size=8, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 100
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))

        self.mission = "use the key to open the door and then get to the goal"


class DoorKeyR2LEnv(DoorKeyEnv):
    def __init__(
        self,
        size=8,
        max_steps: int | None = None,
        **kwargs,
    ):
        super().__init__(
            size=size,
            max_steps=max_steps,
            **kwargs
        )

        # turn_left, turn_right, move, toggle, pickup, RC_get
        self.action_space = gymnasium.spaces.Discrete(6)

        # front_is_clear, left_is_clear, right_is_clear, 
        # goal_on_left, goal_on_right, goal_present, 
        # front_is_lava, front_is_closed_door, front_is_locked_door, 
        # front_is_key, has_key
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(11, ))

    def step(self, action):
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
            # toggle
            real_action = self.actions.toggle
        elif action == 4:
            # pickup
            real_action = self.actions.pickup
        elif action == 5:
            policy_path = "MiniGrid-RandomCrossingR2LS11N5-v0/MiniGrid-RandomCrossingR2LS11N5-v0_seed_104_length_100.pt"
            normalizer_path = policy_path.replace(".pt", ".pkl")
            policy = torch.load(policy_path)
            policy.load_normalizer_param(path=normalizer_path)
            policy.init_hidden_state()
            state = torch.Tensor(self.public_get_abs_obs()[:6])
            action = policy(state, deterministic=True)
            if action == 0:
                real_action = self.actions.left
            elif action == 1:
                real_action = self.actions.right
            elif action == 2:
                real_action = self.actions.forward
            else:
                assert False
        else:
            assert False
        _, rwd, terminated, truncated, info = DoorKeyEnv.step(self, real_action)
        return self.public_get_abs_obs(), rwd, terminated, truncated, info

    def public_get_abs_obs(self):
        b_front_is_clear = self.front_is_clear()
        b_left_is_clear = self.left_is_clear()
        b_right_is_clear = self.right_is_clear()
        b_goal_on_left = self.goal_on_left()
        b_goal_on_right = self.goal_on_right()
        b_goal_present = self.goal_present()
        b_front_is_lava = self.front_is_lava()
        b_front_is_closed_door = self.front_is_closed_door()
        b_front_is_locked_door = self.front_is_locked_door()
        b_front_is_key = self.front_is_key()
        b_has_key = self.has_key()

        obs = [
            float(b_front_is_clear),
            float(b_left_is_clear),
            float(b_right_is_clear),
            float(b_goal_on_left),
            float(b_goal_on_right),
            float(b_goal_present),
            float(b_front_is_lava),
            float(b_front_is_closed_door),
            float(b_front_is_locked_door),
            float(b_front_is_key),
            float(b_has_key),
        ]

        return np.array(obs)

    def reset(self, *, seed: int | None = None, options=None):
        DoorKeyEnv.reset(self, seed=seed, options=options)
        return self.public_get_abs_obs()