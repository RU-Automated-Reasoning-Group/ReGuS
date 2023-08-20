from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from program_env.ProgramEnv import ProgramEnv

import spinup.algos.pytorch.ppo.core as core
from spinup.algos.pytorch.ppo.ppo import ppo


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=999)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--print_program', dest='print_program', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # task = 'cleanHouse'
    # task = 'harvester'
    # task = 'randomMaze'
    # task = 'fourCorners'
    task = 'stairClimber'
    # task = 'topOff'

    def env_fn():
        return ProgramEnv(task=task, seed=args.seed, batch_size=10, print_program=args.print_program)

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    ppo(env_fn=env_fn, actor_critic=core.MLPMaskedActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs, 
        max_ep_len=50, clip_ratio=0.1, target_kl=0.01, train_pi_iters=16, train_v_iters=16)

    
# def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
#         steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
#         vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
#         target_kl=0.01, logger_kwargs=dict(), save_freq=10):