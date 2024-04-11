from itertools import chain
import highway_env
highway_env.register_highway_envs()

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

from highway_general_self.highway_gym import HighwayEnv
from karel.karel_env import KarelEnv

from ant.ant_program_env import AntProgramEnv
from ant.configs import get_configs

from multiworld.envs.mujoco import register_custom_envs as register_mujoco_envs

from collections import namedtuple
from stable_baselines3 import DQN, PPO
import matplotlib.pyplot as plt
import os

import pdb

# all primitive policies
ANT_LOW_TORQUE_MODELS = []
for direction in ['up', 'down', 'left', 'right']:
    filename = os.getcwd() + '/primitives/ant_low_torque/' + direction + '.pt'
    ANT_LOW_TORQUE_MODELS.append(torch.load(filename))

def get_highway_rew(args, env):
    seed = args.seed

    model = PPO.load(args.model_path, env=env)
    rew_list = []
    frame_len_list = []
    success_list = []

    for epoch in range(100):
        done = False
        obs, info = env.reset(seed * (epoch+1))
        rew_coll = 0
        all_rewards = []
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            rew_coll += reward
            all_rewards.append(reward)
        print('seed {}: epoch {} with success {} rew {}/{}'.format(seed * (epoch+1), epoch, rew_coll!=-1, rew_coll, env.steps))
        rew_list.append(rew_coll)
        frame_len_list.append(env.steps)
        success_list.append(rew_coll!=-1)

    print('success rate {} avg rew: {}'.format(np.mean(success_list), np.mean(rew_list)))
    env.close()

def get_karel_rew(args, env):
    seed = args.seed

    model = PPO.load(args.model_path, env=env)
    rew_list = []
    frame_len_list = []
    success_list = []

    for epoch in range(100):
        done = False
        obs, info = env.reset(seed * (epoch+1))
        rew_coll = 0
        all_rewards = []
        all_actions = []
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            all_rewards.append(reward)
            all_actions.append(action)
            rew_coll += reward
        print('seed {}: epoch {} with success {} rew {}/{}'.format(seed * (epoch+1), epoch, not done, np.max(all_rewards), rew_coll))
        rew_list.append(np.max(all_rewards))
        success_list.append(np.max(all_rewards)==1)

    print('success rate {} avg rew: {}'.format(np.mean(success_list), np.mean(rew_list)))
    env.close()

def get_ant_rew(args, env):
    seed = args.seed

    model = PPO.load(args.model_path, env=env)
    rew_list = []
    frame_len_list = []
    success_list = []

    for epoch in range(100):
        done = False
        obs, info = env.reset()
        rew_coll = 0
        all_rewards = []
        all_actions = []
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            all_rewards.append(reward)
            all_actions.append(action)
            rew_coll += reward
        print('seed {}: epoch {} with success {} rew {}/{}'.format(seed * (epoch+1), epoch, not done, np.max(all_rewards), rew_coll))
        rew_list.append(np.max(all_rewards))
        success_list.append(np.max(all_rewards)==1)

    print('success rate {} avg rew: {}'.format(np.mean(success_list), np.mean(rew_list)))
    env.close()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='highway')
    parser.add_argument('--model_path', type=str, default='data/highway_dqn/model')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.env_name in ['doorkey', 'seeder', 'harvester', 'cleanHouse', \
                         'randomMaze', 'stairClimber', 'topOff', 'fourCorners']:
        env = KarelEnv(task=args.env_name, obs_type='raw')
        get_karel_rew(args, env)

    elif args.env_name in ['highway']:
        env = HighwayEnv()
        get_highway_rew(args, env)

    elif args.env_name in ['AntU', 'AntFb', 'AntFg', 'AntMaze']:
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

        # visualize
        get_ant_rew(args, ant_program_env)

    else:
        raise NotImplementedError