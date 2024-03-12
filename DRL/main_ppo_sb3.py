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
import math

from highway_general_self.highway_gym import HighwayEnv
from karel.karel_env import KarelEnv


from custom_net import CustomCNN

from collections import namedtuple
from stable_baselines3 import DQN, PPO

import pdb

def highway_policy(args):
    for exp_id in range(args.num_exps):

        # prepare argument
        args.seed = 1000 * exp_id
        args.store_path = 'data/highway_ppo_seed{}'.format(args.seed)
        if not os.path.exists(args.store_path):
            os.makedirs(args.store_path)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # define environment
        env = HighwayEnv()

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
        model.learn(total_timesteps=int(args.timesteps))
        model.save(os.path.join(args.store_path, 'model'))

def karel_policy(args):
    for exp_id in range(args.num_exps):

        # prepare argument
        args.seed = 1000 * exp_id
        args.store_path = 'data/karel_ppo_{}_seed{}'.format(args.env_name, args.seed)
        if not os.path.exists(args.store_path):
            os.makedirs(args.store_path)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # define environment
        env = KarelEnv(task=args.env_name, obs_type='raw')

        # Create the model
        batch_size = 64
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )

        model = PPO("CnnPolicy",
                    env,
                    policy_kwargs = policy_kwargs,
                    n_steps=batch_size * 12,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=1,
                    tensorboard_log=args.store_path)
        
        # Train the agent
        model.learn(total_timesteps=int(args.timesteps))
        model.save(os.path.join(args.store_path, 'model'))

def ant_policy(args):
    # all primitive policies
    ANT_LOW_TORQUE_MODELS = []
    for direction in ['up', 'down', 'left', 'right']:
        filename = os.getcwd() + '/primitives/ant_low_torque/' + direction + '.pt'
        ANT_LOW_TORQUE_MODELS.append(torch.load(filename))

    for exp_id in range(args.num_exps):

        # prepare argument
        args.seed = 1000 * exp_id
        args.store_path = 'data/ant_ppo_{}_seed{}'.format(args.env_name, args.seed)
        if not os.path.exists(args.store_path):
            os.makedirs(args.store_path)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # evaluate environment
        # create gym environment
        task = args.env_name
        if task == 'AntU':
            env = gym.make('AntULongTestEnv-v0', disable_env_checker=True)
        elif task == 'AntFb':
            env = gym.make('AntFbMedTestEnv-v1', disable_env_checker=True)
        elif task == 'AntMaze':
            env = gym.make('AntMazeMedTestEnv-v1', disable_env_checker=True)
        elif task == 'AntFg':
            env = gym.make('AntFgMedTestEnv-v1', disable_env_checker=True)

        config_dict = get_configs(task)
        config_dict['distance_threshold']['front'] = 1.2
        if task == 'AntMaze':
            config_dict['max_episode_length'] = 1000
        ant_program_env = AntProgramEnv(
            env=env,
            models=ANT_LOW_TORQUE_MODELS,
            goal_threshold=0.5,
            goal_detection_threshold=4.0,
            **config_dict,
        )

        # Create the model
        batch_size = 64
        model = PPO("MlpPolicy",
                    ant_program_env,
                    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 12,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=1,
                    tensorboard_log=args.store_path)
        
        # Train the agent
        model.learn(total_timesteps=int(args.timesteps))
        model.save(os.path.join(args.store_path, 'model'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--num_exps', type=int, default=1)
    parser.add_argument('--env_name', type=str, default='AntU')
    parser.add_argument('--timesteps', type=int, default=1e8)
    parser.add_argument('--store_path', type=str, default='data/ant_ppo_antU_seed0')
    args = parser.parse_args()

    if args.env_name in ['doorkey', 'seeder', 'harvester', 'cleanHouse', \
                         'randomMaze', 'stairClimber', 'topOff', 'fourCorners']:
        karel_policy(args)
    elif args.env_name in ['highway']:
        highway_policy(args)
    elif args.env_name in ['AntU', 'AntFb', 'AntFg', 'AntMaze']:
        ant_policy(args)
    else:
        raise NotImplementedError