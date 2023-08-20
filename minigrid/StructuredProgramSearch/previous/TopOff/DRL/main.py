import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import spinup.algos.pytorch.ppo.core as core
from spinup.algos.pytorch.ppo.ppo import ppo
#from karel.gym_karel_env import KarelGymEnv
from karel.gym_karel_drl_env import KarelGymDRLEnv
from karel.program import Program


class EnvEncoder(nn.Module):
    def __init__(self, h, w):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                        kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                        kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 512),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        avg_x = torch.zeros(1, x.shape[1])
        for i in range(batch_size):
            avg_x += x[i, :]

        return avg_x / batch_size


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
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # task = 'cleanHouse'
    # task = 'harvester'
    # task = 'randomMaze'
    # task = 'fourCorners'
    # task = 'stairClimber'
    task = 'topOff'

    # make your life easier first
    program = Program()

    def env_fn():
        #return KarelGymEnv(task=task, program=program, seed=args.seed, encoder=EnvEncoder(h=12, w=12))
        return KarelGymDRLEnv(task=task, seed=args.seed, encoder=EnvEncoder(h=12, w=12))

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    ppo(env_fn=env_fn, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs, 
        max_ep_len=40, clip_ratio=0.05, target_kl=0.01, train_pi_iters=4, train_v_iters=4,
        pi_lr=1e-3, vf_lr=1e-3)

    
# def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
#         steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
#         vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,12
#         target_kl=0.01, logger_kwargs=dict(), save_freq=10):