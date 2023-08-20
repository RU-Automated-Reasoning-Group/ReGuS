import numpy as np
import gym
from gym import spaces
import torch

from .robot import KarelRobot
from .dsl import k_action
from network.network import ACTION_NAME


class KarelGymEnv(gym.Env):

    def __init__(self, task, seed, encoder) -> None:
        super().__init__()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float)

        self.task = task
        self.seed = seed
        self.encoder = encoder

        self.robot = KarelRobot(task=self.task, seed=self.seed)


    # only for stairClimber and fourCorner
    def step(self, action):
        action = k_action(ACTION_NAME[action.item()])
        reward, done = self.robot.execute_single_action(action)
        observation = self._get_obs()
    
        info = {}
        return observation, reward, done, info

    def others_step(self, action):

        action = k_action(ACTION_NAME[action.item()])
        reward = self.robot.execute_single_action(action)
        observation = self._get_obs()

        done = False
        #if reward == 1 or reward == -1:
        #    done = True
        #else:
        #    reward = 0
        if reward == 1:
            done = True

        info = {}
        return observation, reward, done, info

    def _get_obs(self):
        robot_state = torch.stack([torch.from_numpy(self.robot.get_state())], dim=0)
        obs = self.encoder(robot_state.float()).detach().numpy()
        return obs

    def reset(self):
        self.seed += 1
        self.robot = KarelRobot(task=self.task, seed=self.seed)

        return self._get_obs()