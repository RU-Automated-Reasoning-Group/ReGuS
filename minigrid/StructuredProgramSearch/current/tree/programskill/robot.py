import copy

import gym
import gymnasium
import numpy as np

from programskill.push_dsl import k_action, k_cond

# from gym_minigrid.gym_minigrid.wrappers import *


class NegativeReward(Exception):
    pass


# single karel robot
class SkillRobot:
    def __init__(self, task, seed=999, max_steps=200):
        self.task = task
        self.seed = seed
        self.max_steps = max_steps
        self.steps = 0
        self.action_steps = 0

        # self.env = gym.make(self.task, seed=seed)
        self.env = gym.make(self.task, seed=seed)
        # self.env.render()
        # self.env.seed(seed)
        # import pdb

        # for i in range(seed + 1):
        self.env.reset()
        # for i in range(2):
        # pdb.set_trace()
        # self.env.reset()
        # self.env.render("human")
        # self.env.render()

        self.reward = 0

        self.while_start_robot_pos = None
        self.while_moved = False

        self.active = True
        self.force_execution = False
        self.returned = False
        self.reach_goal_times = 0
        self.success = False

    def check_reward(self):
        return self.reward

    def check_success(self):
        return self.success

    def make_step(self):
        self.steps += 1
        if self.steps > 600:
            print("so many steps")

    def no_fuel(self):
        # the self.steps is not used since it tracks the number of iterations the loop is run
        # here we use the actual steps made on the gym env
        return self.max_steps < self.env._elapsed_steps

    def execute_single_cond(self, cond):
        # print(type(cond))
        assert isinstance(cond, k_cond)
        return cond(self.env)

    def execute_single_action(self, action):
        if not isinstance(action, k_action):
            import pdb

            pdb.set_trace()
        assert isinstance(action, k_action)
        # self.reward += action(self.env, self)
        tmp_rwd, success = action(self.env, self)
        self.success = success
        if tmp_rwd == 0.5 and self.reward == 0.0:
            self.reward = 0.5
        elif tmp_rwd == 1.0 and self.reward != -1.0:
            self.reward = 1.0
        elif tmp_rwd == -1.0:
            self.reward = -1.0
        # print(f"self.reward {self.reward}")

    def get_abs_state(self):
        print("block_at_goal", self.env.block_at_goal())
        print("block_is_grasped", self.env.block_is_grasped())
        print("block_above_goal", self.env.block_above_goal())
        print("block_inside_gripper", self.env.block_inside_gripper())
        print("block_below_gripper", self.env.block_below_gripper())
        print("gripper_are_closed", self.env.gripper_are_closed())
        print("gripper_are_open", self.env.gripper_are_open())
