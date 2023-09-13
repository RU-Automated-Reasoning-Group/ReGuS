from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid


class PutNearTwoRoomEnv(RoomGrid):

    """
    ## Description

    The agent has to pick up a box which is placed in another room, behind a
    locked door. This environment can be solved without relying on language.

    ## Mission Space

    "pick up the {color} box"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

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

    1. The agent picks up the correct box.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Unlock-v0`

    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 6
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[],
        )

        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return f"put down"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, obj_pos = self.add_object(1, 0, kind="ball")
        # Make sure the two rooms are directly connected by a locked door
        door, door_pos = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"put down"

        self.target_pos_set = set()
        if obj_pos[0] == door_pos[0] + 1 and obj_pos[1] == door_pos[1]:
            # the object is next to the door
            self.target_pos_set.add((obj_pos[0] - 2, obj_pos[1]))
        else:
            # check around the object
            offset = [-1, 0, 1]
            for off_i in range(0, len(offset)):
                for off_j in range(0, len(offset)):
                    loc = (obj_pos[0] + offset[off_i], obj_pos[1] + offset[off_j])
                    if self.grid.get(*loc) is None:
                        self.target_pos_set.add(loc)
        
        # print(f"target_pos_set: {self.target_pos_set}")
        

    def step(self, action):
        preCarrying = self.carrying

        obs, reward, terminated, truncated, info = super().step(action)

        u, v = self.dir_vec
        ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v) 

        if action == self.actions.drop and preCarrying:
            if self.grid.get(ox, oy) is preCarrying:
                if (ox, oy) in self.target_pos_set:
                    reward = self._reward()
                    terminated = True


        return obs, reward, terminated, truncated, info
