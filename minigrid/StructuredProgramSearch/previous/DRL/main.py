import gym
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from karel.dsl import *
from karel.env import KarelRobotExecutor
from network.network import CodePredictor, EnvEncoder, ValueFunction
from karel.gym_karel_env import KarelGymEnv

import spinup.algos.pytorch.ppo.core as core
from spinup.algos.pytorch.ppo.ppo import ppo
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task = 'stairClimber'

    if task == 'cleanHouse':
        encoder = EnvEncoder(h=14, w=22)
    elif task == 'harvester':
        encoder = EnvEncoder(h=8, w=8)
    elif task == 'randomMaze':
        encoder = EnvEncoder(h=8, w=8)
    elif task == 'fourCorners':
        encoder = EnvEncoder(h=12, w=12)
    elif task == 'stairClimber':
        encoder = EnvEncoder(h=12, w=12)
    elif task == 'topOff':
        encoder = EnvEncoder(h=12, w=12)

    #lambda : gym.make(args.env)
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    ppo(KarelGymEnv(task=task, seed=args.seed, encoder=encoder), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs, 
        max_ep_len=150, clip_ratio=0.2, target_kl=0.01, train_pi_iters=16, train_v_iters=16)

    
# def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
#         steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
#         vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
#         target_kl=0.01, logger_kwargs=dict(), save_freq=10):