from __future__ import annotations

import gymnasium
import numpy as np
import torch

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class LockedRoom:
    def __init__(self, top, size, doorPos):
        self.top = top
        self.size = size
        self.doorPos = doorPos
        self.color = None
        self.locked = False

    def rand_pos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(topX + 1, topX + sizeX - 1, topY + 1, topY + sizeY - 1)


class LockedRoomEnv(MiniGridEnv):

    """
    ## Description

    The environment has six rooms, one of which is locked. The agent receives
    a textual mission string as input, telling it which room to go to in order
    to get the key that opens the locked room. It then has to go into the locked
    room in order to reach the final goal. This environment is extremely
    difficult to solve with vanilla reinforcement learning alone.

    ## Mission Space

    "get the {lockedroom_color} key from the {keyroom_color} room, unlock the {door_color} door and go to the goal"

    {lockedroom_color}, {keyroom_color}, and {door_color} can be "red", "green",
    "blue", "purple", "yellow" or "grey".

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

    - `MiniGrid-LockedRoom-v0`

    """

    def __init__(self, size=19, max_steps: int | None = None, **kwargs):
        self.size = size

        if max_steps is None:
            max_steps = 300
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES] * 3,
        )
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(lockedroom_color: str, keyroom_color: str, door_color: str):
        return (
            f"get the {lockedroom_color} key from the {keyroom_color} room,"
            f" unlock the {door_color} door and go to the goal"
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        for i in range(0, width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height - 1, Wall())
        for j in range(0, height):
            self.grid.set(0, j, Wall())
            self.grid.set(width - 1, j, Wall())

        # Hallway walls
        lWallIdx = width // 2 - 2
        rWallIdx = width // 2 + 2
        for j in range(0, height):
            self.grid.set(lWallIdx, j, Wall())
            self.grid.set(rWallIdx, j, Wall())

        self.rooms = []

        # Room splitting walls
        for n in range(0, 3):
            j = n * (height // 3)
            for i in range(0, lWallIdx):
                self.grid.set(i, j, Wall())
            for i in range(rWallIdx, width):
                self.grid.set(i, j, Wall())

            roomW = lWallIdx + 1
            roomH = height // 3 + 1
            self.rooms.append(LockedRoom((0, j), (roomW, roomH), (lWallIdx, j + 3)))
            self.rooms.append(
                LockedRoom((rWallIdx, j), (roomW, roomH), (rWallIdx, j + 3))
            )

        # Choose one random room to be locked
        lockedRoom = self._rand_elem(self.rooms)
        lockedRoom.locked = True
        goalPos = lockedRoom.rand_pos(self)
        self.grid.set(*goalPos, Goal())

        # Assign the door colors
        colors = set(COLOR_NAMES)
        for room in self.rooms:
            color = self._rand_elem(sorted(colors))
            colors.remove(color)
            room.color = color
            if room.locked:
                self.grid.set(*room.doorPos, Door(color, is_locked=True))
            else:
                self.grid.set(*room.doorPos, Door(color))

        # Select a random room to contain the key
        while True:
            keyRoom = self._rand_elem(self.rooms)
            if keyRoom != lockedRoom:
                break
        keyPos = keyRoom.rand_pos(self)
        self.grid.set(*keyPos, Key(lockedRoom.color))

        # Randomize the player start position and orientation
        self.agent_pos = self.place_agent(
            top=(lWallIdx, 0), size=(rWallIdx - lWallIdx, height)
        )

        # Generate the mission string
        self.mission = (
            "get the %s key from the %s room, "
            "unlock the %s door and "
            "go to the goal"
        ) % (lockedRoom.color, keyRoom.color, lockedRoom.color)

class LockedRoomR2LEnv(LockedRoomEnv):
    def __init__(
        self,
        size=19,
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
        _, rwd, terminated, truncated, info = LockedRoomEnv.step(self, real_action)
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
        LockedRoomEnv.reset(self, seed=seed, options=options)
        return self.public_get_abs_obs()