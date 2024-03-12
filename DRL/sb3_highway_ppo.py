from itertools import chain
import highway_env
highway_env.register_highway_envs()

import gymnasium as gym

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import os

from highway_general.robot_perc_warper import HighwayRobot

import spinup.algos.pytorch.progs.core as core

from collections import namedtuple
from stable_baselines3 import DQN, PPO

import pdb

def mlp_policy():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--store_path', type=str, default='data/highway_dqn')
    args = parser.parse_args()

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # define program
    prog = core.get_gt_prog()

    # define environment
    env = gym.make('highway-fast-v0')
    # env = gym.make('highway-v0')
    env = HighwayRobot(env, prog, act_dim=3, cont_act=True)

    # Create the model
    batch_size = 64
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.8,
                verbose=2,
                tensorboard_log=args.store_path)
    
    # Train the agent
    model.learn(total_timesteps=int(2e4))
    model.save(os.path.join(args.store_path, 'model'))

if __name__ == "__main__":
    mlp_policy()