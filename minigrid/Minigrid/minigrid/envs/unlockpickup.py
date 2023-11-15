from __future__ import annotations

import gymnasium
import numpy as np
import torch

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid


class UnlockPickupEnv(RoomGrid):

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
            ordered_placeholders=[COLOR_NAMES],
        )

        if max_steps is None:
            max_steps = 100

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str):
        return f"pick up the {color} box"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="ball")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                print("get reward")
                terminated = True

        return obs, reward, terminated, truncated, info

class UnlockPickupR2LEnv(UnlockPickupEnv):
    def __init__(
        self,
        max_steps: int | None = None,
        **kwargs,
    ):
        super().__init__(
            max_steps=max_steps,
            **kwargs
        )

        # turn_left, turn_right, move, toggle, pickup, drop, RC_get
        self.action_space = gymnasium.spaces.Discrete(7)

        # front_is_clear, left_is_clear, right_is_clear, 
        # goal_on_left, goal_on_right, goal_present, 
        # front_is_lava, front_is_closed_door, front_is_locked_door, 
        # front_is_key, has_key, front_is_ball
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(12, ))

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
            # drop
            real_action = self.actions.drop
        elif action == 6:
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
        _, rwd, terminated, truncated, info = UnlockPickupEnv.step(self, real_action)
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
        b_front_is_ball = self.front_is_ball()

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
            float(b_front_is_ball),
        ]

        return np.array(obs)

    def reset(self, *, seed: int | None = None, options=None):
        UnlockPickupEnv.reset(self, seed=seed, options=options)
        return self.public_get_abs_obs()