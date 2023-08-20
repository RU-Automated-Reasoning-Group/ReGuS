from asyncio import tasks
import numpy as np
import gym
from gym import spaces
import torch

from .robot import KarelRobot
from .dsl import k_action

ACTION_NAME = [
    'noop',
    'move',
    'turn_right',
    'turn_left',
    'pick_marker',
    'put_marker'
]

class KarelGymEnv(gym.Env):

    def __init__(self, task, seed) -> None:
        super().__init__()

        #self.action_space = spaces.Discrete(5)
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float)

        # NOTE: try to make your life easier, try continuous first
        
        self.lines, self.act_dim = 5, 6
        
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.lines * self.act_dim,), dtype=np.float)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12 * 12 * 16,), dtype=np.float)

        self.task = task
        self.seed = seed

        self.robot = KarelRobot(task=self.task, seed=self.seed)

    # NOTE: input is a sequence of actions
    def step_discrete(self, action):

        # multiple lines of code
        for a in action:
            a = a.item()
            if a != 0 and not self.robot.no_fuel():
                act = k_action(ACTION_NAME[a])
                reward = self.robot.execute_single_action(act)
                self.robot.steps += 1

        observation = self._get_obs()
        done = True if self.robot.no_fuel() or reward == 2 else False
        info = {}

        return observation, reward, done, info

    # NOTE: input is the action matrix
    def step(self, action):

        action = action.reshape(self.lines, self.act_dim)

        return self.step_discrete(action.argmax(dim=1))

    def _get_obs(self):
        
        robot_state = torch.stack([torch.from_numpy(self.robot.get_state())], dim=0)

        return robot_state.flatten(start_dim=1)
    
    def reset(self):
        self.seed += 1
        self.robot = KarelRobot(task=self.task, seed=self.seed)

        return self._get_obs()
