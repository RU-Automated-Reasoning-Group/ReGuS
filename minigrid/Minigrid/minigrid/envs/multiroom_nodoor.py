from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

import gymnasium
import torch
import numpy as np

class MultiRoom:
    def __init__(self, top, size, entryDoorPos, exitDoorPos):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos


class MultiRoomNoDoorEnv(MiniGridEnv):

    """
    ## Description

    This environment has a series of connected rooms with doors that must be
    opened in order to get to the next room. The final room has the green goal
    square the agent must get to. This environment is extremely difficult to
    solve using RL alone. However, by gradually increasing the number of rooms
    and building a curriculum, the environment can be solved.

    ## Mission Space

    "traverse the rooms to get to the goal"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
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

    S: size of map SxS.
    N: number of rooms.

    - `MiniGrid-MultiRoom-N2-S4-v0` (two small rooms)
    - `MiniGrid-MultiRoom-N4-S5-v0` (four rooms)
    - `MiniGrid-MultiRoom-N6-v0` (six rooms)

    """

    def __init__(
        self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        max_steps: int | None = None,
        **kwargs,
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize

        self.rooms = []

        mission_space = MissionSpace(mission_func=self._gen_mission)

        self.size = 25

        if max_steps is None:
            max_steps = 150

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "traverse the rooms to get to the goal"

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms + 1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (self._rand_int(0, width - 2), self._rand_int(0, width - 2))

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos,
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):
            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                pass
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # # Note: the use of sorting here guarantees determinism,
                # # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                # entryDoor = Door(doorColor)
                self.grid.set(room.entryDoorPos[0], room.entryDoorPos[1], None)
                prevDoorColor = doorColor

                prevRoom = roomList[idx - 1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = "traverse the rooms to get to the goal"

    def _placeRoom(self, numLeft, roomList, minSz, maxSz, entryDoorWall, entryDoorPos):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz + 1)
        sizeY = self._rand_int(minSz, maxSz + 1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = (
                topX + sizeX < room.top[0]
                or room.top[0] + room.size[0] <= topX
                or topY + sizeY < room.top[1]
                or room.top[1] + room.size[1] <= topY
            )

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(MultiRoom((topX, topY), (sizeX, sizeY), entryDoorPos, None))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):
            # Pick which wall to place the out door on
            wallSet = {0, 1, 2, 3}
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (topX + sizeX - 1, topY + self._rand_int(1, sizeY - 1))
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY + sizeY - 1)
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (topX, topY + self._rand_int(1, sizeY - 1))
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY)
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos,
            )

            if success:
                break

        return True

class MultiRoomNoDoorR2LEnv(MultiRoomNoDoorEnv):
    def __init__(
        self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        max_steps: int | None = None,
        **kwargs,
    ):
        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=maxRoomSize,
            max_steps=max_steps,
            **kwargs
        )

        # turn_left, turn_right, move, RC_get
        self.action_space = gymnasium.spaces.Discrete(4)

        # front_is_clear, left_is_clear, right_is_clear, goal_on_left, goal_on_right, goal_present, front_is_lava
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7, ))

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
            policy_path = "MiniGrid-RandomCrossingR2LS11N5-v0/MiniGrid-RandomCrossingR2LS11N5-v0_seed_104_length_100.pt"
            normalizer_path = policy_path.replace(".pt", ".pkl")
            policy = torch.load(policy_path)
            policy.load_normalizer_param(path=normalizer_path)
            policy.init_hidden_state()
            state = torch.Tensor(self.public_get_abs_obs()[:-1])
            real_action = policy(state, deterministic=True) 
        else:
            assert False
        _, rwd, terminated, truncated, info = MultiRoomNoDoorEnv.step(self, real_action)
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
        MultiRoomNoDoorEnv.reset(self, seed=seed, options=options)
        return self.public_get_abs_obs()