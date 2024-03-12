import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

def vis_train():
    # path = 'data/ppo_highway_attn_tog_0/ppo_highway_attn_tog_0_s0/progress.txt'
    # path = 'data/ppo_highway_attn_tog_ent_0/ppo_highway_attn_tog_ent_0_s0/progress.txt'
    path = 'data/dqn_highway_attn_0/dqn_highway_attn_0_s0/progress.txt'
    # path = 'data/ppo_highway_attn_keep_gtwith_0/ppo_highway_attn_keep_gtwith_0_s0/progress.txt'
    # path = 'data/ppo_highway_attn_keep_gtwith_2_0/ppo_highway_attn_keep_gtwith_2_0_s0/progress.txt'
    # path = 'data/ppo_highway_attn_comb_2_0/ppo_highway_attn_comb_2_0_s0/progress.txt'
    # path = 'data/ppo_highway_attn_keep_2_0/ppo_highway_attn_keep_2_0_s0/progress.txt'
    # path = 'data/ppo_highway_2_0/ppo_highway_2_0_s0/progress.txt'
    # path = 'data/ppo_highway_attn_2_0/ppo_highway_attn_2_0_s0/progress.txt'
    # path = 'data/ppo_highway_attn_keep_0/ppo_highway_attn_keep_0_s0/progress.txt'
    # path = 'data/ppo_highway_0/ppo_highway_0_s0/progress.txt'
    # path = 'data/ppo_highway_attn_0/ppo_highway_attn_0_s0/progress.txt'
    # pdb.set_trace()
    df = pd.read_csv(path, delimiter='\t')

    derror = np.array(df['DError'])
    # rew = np.array(df['AverageEpRet'])
    # rew = np.array(df['AverageEpLen'])
    # pdb.set_trace()
    plt.figure()
    plt.plot(np.arange(len(derror)), derror)
    # plt.plot(np.arange(len(rew)), rew)
    # plt.savefig('results/ppo0_rew.png')
    # plt.savefig('results/ppo_highway_attn_keep_2.png')
    # plt.savefig('results/ppo_highway_attn_keep.png')
    # plt.savefig('results/ppo_highway_attn.png')
    # plt.savefig('results/ppo_highway_2.png')
    # plt.savefig('results/ppo0_attn_rew.png')
    # plt.savefig('results/ppo0_attn_len.png')
    # plt.savefig('results/ppo_highway_gtwith.png')
    # plt.savefig('results/ppo_highway_gtwith_2.png')
    # plt.savefig('results/ppo_highway_attn_comb_2.png')
    # plt.savefig('results/dqn_highway_attn_0.png')
    # plt.savefig('results/ppo_highway_attn_tog_ent.png')
    plt.savefig('results/dqn_highway_error.png')

    # last_seeds = [seed_data.count(',') for seed_data in df['LastSeed']]
    # pdb.set_trace()
    # plt.figure()
    # plt.plot(np.arange(len(last_seeds)), last_seeds)
    # plt.savefig('results/ppo_highway_attn_keep_2_fail_num.png')
    # plt.savefig('results/ppo_highway_gtwith_fail_num.png')
    # plt.savefig('results/ppo_highway_gtwith_2_fail_num.png')

    # loss_v = np.array(df['LossV'])
    # loss_pi = np.array(df['LossPi'])
    # plt.figure()
    # plt.plot(np.arange(len(loss_v)), loss_v)
    # plt.savefig('results/ppo_highway_attn_keep_losspi.png')
    # plt.savefig('results/ppo_highway_attn_comb_2_lossv.png')
    # plt.savefig('results/ppo_highway_attn_comb_2_losspi.png')

def vis_rew():
    # path = 'data/reward_old.npy'
    # path = 'data/reward_comb.npy'
    # path = 'data/reward_comb_tog.npy'
    # path = 'racetrack_ppo/rew.npy'
    # path = 'highway_general/scripts/racetrack_ppo/rew.npy'
    path = 'racetrack_dqn_cnn/rew.npy'
    rew_data = np.load(path)

    plt.figure()
    plt.hist(rew_data)
    # plt.savefig('data/figs/rew_old_hist.png')
    # plt.savefig('data/figs/rew_comb_hist.png')
    # plt.savefig('data/figs/rew_tog_hist.png')
    # plt.savefig('data/figs/reward_dqn_hist.png')
    # plt.savefig('data/figs/reward_dqn_sb3_hist.png')
    plt.savefig('data/figs/reward_dqn_cnn_sb3_hist.png')

def vis_rew_tensorboard():
    path = 'data/karel_ppo_doorkey/PPO_2/data.csv'
    df = pd.read_csv(path, header=0)
    steps = df['Step']
    value = df['Value']

    plt.figure()
    plt.plot(steps, value)
    plt.savefig('store/figs/doorkey.pdf')

def vis_rew_seeder():
    path = 'data/karel_ppo/PPO_3/data.csv'
    df = pd.read_csv(path, header=0)
    steps = df['Step']
    value = df['Value']

    n = np.sqrt(value * 72 + 0.25) - 0.5
    reward = n / 36

    plt.figure()
    plt.plot(steps, reward)
    plt.savefig('store/figs/seeder.pdf')

def vis_rew_doorkey():
    path = 'data/karel_ppo_doorkey/PPO_2/data.csv'
    df = pd.read_csv(path, header=0)
    steps = df['Step']
    value = df['Value']

    pdb.set_trace()

    plt.figure()
    plt.plot(steps, value)
    plt.savefig('store/figs/doorkey.pdf')


if __name__ == '__main__':
    # vis_train()
    # vis_rew()
    # vis_rew_tensorboard()
    # vis_rew_seeder()
    vis_rew_doorkey()