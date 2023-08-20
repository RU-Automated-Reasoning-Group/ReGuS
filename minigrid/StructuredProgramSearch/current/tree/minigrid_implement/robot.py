import copy

import gymnasium as gym
import numpy as np

from minigrid_implement.dsl import k_action, k_cond


class NegativeReward(Exception):
    pass


# single minigrid robot
class MiniGridRobot:
    def __init__(self, task, seed=999, max_steps=50, make_video=False):
        if task.startswith("MiniGrid-MultiRoom"):
            max_steps = 150
        elif (
            task == "MiniGrid-RandomCrossingS11N5-v0"
            or task == "MiniGrid-RandomLavaCrossingS11N5-v0"
        ):
            max_steps = 100
        elif task == "MiniGrid-LockedRoom-v0":
            max_steps = 600
        elif task.startswith("MiniGrid-KeyCorridor"):
            max_steps = 400
        elif task.startswith("MiniGrid-DoorKey"):
            max_steps = 100
        elif task == "MiniGrid-UnlockPickup-v0":
            max_steps = 120
        self.task = task
        self.seed = seed
        self.max_steps = max_steps
        self.steps = 0
        self.action_steps = 0

        # if make_video:
        #     self.env = gym.make(self.task, render_mode="human")
        # else:
        self.env = gym.make(
            self.task,
            render_mode="rgb_array",
        )
        self.env.reset(seed=seed)

        self.reward = 0

        self.initial_pos = np.array(copy.deepcopy(self.env.agent_pos))

        self.active = True
        self.force_execution = False
        self.returned = False
        self.reach_goal_times = 0

    def render(self, dir=None):
        self.env.env.env.render(dir)

    def update_reward(self, reward_in):
        self.reward += reward_in

    def check_reward(self):
        # if self.returned and (self.reach_goal_times == 1):
        # return self.env.get_reward()
        # return 0.0
        return self.env.get_reward()

    def make_step(self):
        self.steps += 1
        if self.steps > 600:
            print("so many steps")

    def no_fuel(self):
        return self.max_steps < self.steps

    def execute_single_cond(self, cond):
        assert isinstance(cond, k_cond)
        return cond(self.env)

    def execute_single_action(self, action):
        if not isinstance(action, k_action):
            import pdb

            pdb.set_trace()
        assert isinstance(action, k_action)
        return action(self.env, self)
