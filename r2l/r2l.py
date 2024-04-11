import os

import torch
import hashlib
from collections import OrderedDict

from util.env import env_factory, eval_policy, interactive_eval
from util.logo import print_logo

def create_args(args):
  # default
  args.arch = 'gru'
  args.layers = "64"
  args.batch_size = 6
  args.num_steps = 1e4
  args.prenormalize_steps = 100
  args.timesteps = 1e8

  # karel
  if args.env == 'seeder':
    args.env = 'karel-seeder-{}'.format(args.seed)
  elif args.env == 'doorkey':
    args.env = 'karel-doorkey-{}'.format(args.seed)
  elif args.env == 'harvester':
    args.env = 'karel-harvester-{}'.format(args.seed)
  elif args.env == 'cleanHouse':
    args.env = 'karel-cleanHouse-{}'.format(args.seed)
  elif args.env == 'randomMaze':
    args.env = 'karel-randomMaze-{}'.format(args.seed)
  elif args.env == 'stairClimber':
    args.env = 'karel-stairClimber-{}'.format(args.seed)
  elif args.env == 'topOff':
    args.env = 'karel-topOff-{}'.format(args.seed)
  elif args.env == 'fourCorners':
    args.env = 'karel-fourCorners-{}'.format(args.seed)
  # highway
  elif args.env == 'highway':
    args.env = 'highway-v0-{}'.format(args.seed)
  # ant
  elif args.env == 'AntU':
    args.env = 'AntU-{}'.format(args.seed)
  elif args.env == 'AntFb':
    args.env = 'AntFb-{}'.format(args.seed)
  elif args.env == 'AntFg':
    args.env = 'AntFg-{}'.format(args.seed)
  elif args.env == 'AntMaze':
    args.env = 'AntMaze-{}'.format(args.seed)


  args.save_actor = 'store/{}.pt'.format(args.env)

  return args


if __name__ == "__main__":
  import sys, argparse, time, os
  parser = argparse.ArgumentParser()
  print_logo(subtitle="Recurrent Reinforcement Learning for Robotics.")

  if len(sys.argv) < 2:
    print("Usage: python apex.py [option]", sys.argv)
    exit(1)

  # Options common to all RL algorithms.
  parser.add_argument("--nolog",                  action='store_true')              # store log data or not.
  parser.add_argument("--arch",           "-r",   default='ff')                     # either ff, lstm, or gru
  parser.add_argument("--seed",           "-s",   default=0,           type=int)    # random seed for reproducibility
  parser.add_argument("--traj_len",       "-tl",  default=1000,        type=int)    # max trajectory length for environment
  parser.add_argument("--env",            "-e",   default="Hopper-v3", type=str)    # environment to train on
  parser.add_argument("--layers",                 default="256,256",   type=str)    # hidden layer sizes in policy
  parser.add_argument("--timesteps",      "-t",   default=1e6,         type=float)  # timesteps to explore environment for

  if sys.argv[1] == 'train':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Proximal Policy Optimization.

    """
    from algos.ppo import run_experiment
    parser.add_argument("--prenormalize_steps",     default=10000,         type=int)      
    parser.add_argument("--num_steps",              default=5000,          type=int)      

    parser.add_argument('--discount',               default=0.99,          type=float)    # the discount factor
    parser.add_argument('--std',                    default=0.13,          type=float)    # the fixed exploration std
    parser.add_argument("--a_lr",           "-alr", default=1e-4,          type=float)    # adam learning rate for actor
    parser.add_argument("--c_lr",           "-clr", default=1e-4,          type=float)    # adam learning rate for critic
    parser.add_argument("--eps",            "-ep",  default=1e-6,          type=float)    # adam eps
    parser.add_argument("--kl",                     default=0.02,          type=float)    # kl abort threshold
    parser.add_argument("--entropy_coeff",          default=0.0,           type=float)
    parser.add_argument("--grad_clip",              default=0.05,          type=float)
    parser.add_argument("--batch_size",             default=64,            type=int)      # batch size for policy update
    parser.add_argument("--epochs",                 default=3,             type=int)      # number of updates per iter
    parser.add_argument("--mirror",                 default=0,             type=float)
    parser.add_argument("--sparsity",               default=0,             type=float)

    parser.add_argument("--save_actor",             default=None,          type=str)
    parser.add_argument("--save_critic",            default=None,          type=str)
    parser.add_argument("--workers",                default=4,             type=int)
    parser.add_argument("--redis",                  default=None,          type=str)

    parser.add_argument("--num_exps", default=1, type=int)

    parser.add_argument("--logdir",                 default="./logs/ppo/", type=str)
    args = parser.parse_args()

    # fix arguments
    for exp_id in range(args.num_exps):
      print('current {} experiment'.format(exp_id))
      args.seed = 1000 * exp_id
      args = create_args(args)
      run_experiment(args)

  elif sys.argv[1] == 'eval':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Proximal Policy Optimization.

    """
    from algos.ppo_evaluate import run_experiment
    parser.add_argument("--prenormalize_steps",     default=10000,         type=int)      
    parser.add_argument("--num_steps",              default=5000,          type=int)      
    parser.add_argument("--policy", "-p", default=None,         type=str)
    parser.add_argument("--eval_num", default=100, type=int)

    parser.add_argument('--discount',               default=0.99,          type=float)    # the discount factor
    parser.add_argument('--std',                    default=0.13,          type=float)    # the fixed exploration std
    parser.add_argument("--a_lr",           "-alr", default=1e-4,          type=float)    # adam learning rate for actor
    parser.add_argument("--c_lr",           "-clr", default=1e-4,          type=float)    # adam learning rate for critic
    parser.add_argument("--eps",            "-ep",  default=1e-6,          type=float)    # adam eps
    parser.add_argument("--kl",                     default=0.02,          type=float)    # kl abort threshold
    parser.add_argument("--entropy_coeff",          default=0.0,           type=float)
    parser.add_argument("--grad_clip",              default=0.05,          type=float)
    parser.add_argument("--batch_size",             default=64,            type=int)      # batch size for policy update
    parser.add_argument("--epochs",                 default=3,             type=int)      # number of updates per iter
    parser.add_argument("--mirror",                 default=0,             type=float)
    parser.add_argument("--sparsity",               default=0,             type=float)

    parser.add_argument("--save_actor",             default=None,          type=str)
    parser.add_argument("--save_critic",            default=None,          type=str)
    parser.add_argument("--workers",                default=4,             type=int)
    parser.add_argument("--redis",                  default=None,          type=str)

    parser.add_argument("--logdir",                 default="./logs/ppo/", type=str)
    args = parser.parse_args()

    # fix arguments
    args = create_args(args)

    run_experiment(args)

  else:
    print("Invalid option '{}'".format(sys.argv[1]))
