from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import numpy as np
import os

import pdb

def var_plot(args, all_time_steps, all_rewards, all_labels, fig_path):
    max_time_steps = {}
    entire_time_steps = []
    style = ['r-', 'k-', 'b-', 'y-']

    # get max time steps
    for label in all_labels:
        for time_steps in all_time_steps[label]:
            if label not in max_time_steps:
                max_time_steps[label] = time_steps[-1]
            else:
                max_time_steps[label] = max(max_time_steps[label], time_steps[-1])
            entire_time_steps += time_steps

    # clean reward
    entire_time_steps = np.array(sorted(list(set(entire_time_steps))))
    clean_all_rewards = {label:[] for label in all_labels}
    for label in all_labels:
        for time_step, reward in zip(all_time_steps[label], all_rewards[label]):
            cur_time_step_id = 0
            new_reward = []
            for need_step in entire_time_steps:
                # next
                if cur_time_step_id < len(time_step) and need_step > time_step[cur_time_step_id]:
                    cur_time_step_id += 1
                while cur_time_step_id < len(time_step) and need_step > time_step[cur_time_step_id]:
                    assert time_step[cur_time_step_id-1] == time_step[cur_time_step_id]
                    cur_time_step_id += 1
                # store
                if cur_time_step_id >= len(time_step):
                    new_reward.append(reward[-1])
                else:
                    new_reward.append(reward[cur_time_step_id])
            clean_all_rewards[label].append(new_reward)

    # plot
    entire_time_steps = np.array(entire_time_steps)
    if args.end_timestep is not None:
        last_id = np.argmax(entire_time_steps > args.end_timestep)
    

    plt.figure()
    for label_id, label in enumerate(all_labels):
        cur_all_rewards = np.array(clean_all_rewards[label])
        if args.end_timestep is None:
            plt.plot(entire_time_steps, np.mean(cur_all_rewards, axis=0), style[label_id], label=label)
            plt.fill_between(entire_time_steps, \
                            np.min(cur_all_rewards, axis=0), \
                            np.max(cur_all_rewards, axis=0), \
                            alpha=0.2, edgecolor=style[label_id][0], facecolor=style[label_id][0])
        else:
            plt.plot(entire_time_steps[:last_id], np.mean(cur_all_rewards[:, :last_id], axis=0), style[label_id], label=label)
            plt.fill_between(entire_time_steps[:last_id], \
                            np.min(cur_all_rewards[:, :last_id], axis=0), \
                            np.max(cur_all_rewards[:, :last_id], axis=0), \
                            alpha=0.2, edgecolor=style[label_id][0], facecolor=style[label_id][0])

    plt.legend(fontsize=14)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(fig_path)


def load_data_raw(DRL_paths, DRL_abs_paths, Regus_path):
    # init
    all_labels = ['DRL', 'DRL-abs', 'ReGus']
    all_time_steps = {'DRL':[], 'DRL-abs':[], 'ReGus':[]}
    all_rewards = {'DRL':[], 'DRL-abs':[], 'ReGus':[]}

    # load raw data from DRL_paths
    for path in DRL_paths:
        all_time_steps['DRL'].append([])
        all_rewards['DRL'].append([])
        for each_data in summary_iterator(path):
            if len(each_data.summary.value) > 0 and 'ep_rew_mean' in each_data.summary.value[0].tag:
                all_time_steps['DRL'][-1].append(each_data.step)
                all_rewards['DRL'][-1].append(each_data.summary.value[0].simple_value)

    # load raw data from DRL-abs_paths
    for path in DRL_abs_paths:
        all_time_steps['DRL-abs'].append([])
        all_rewards['DRL-abs'].append([])
        for each_data in summary_iterator(path):
            if len(each_data.summary.value) > 0 and 'return' in each_data.summary.value[0].tag:
                all_time_steps['DRL-abs'][-1].append(each_data.step)
                all_rewards['DRL-abs'][-1].append(each_data.summary.value[0].simple_value)

    # load raw data from ReGus
    regus_data = np.load(Regus_path, allow_pickle=True)
    for time_steps, rewards, _ in regus_data:
        all_time_steps['ReGus'].append(time_steps)
        all_rewards['ReGus'].append(rewards)

    return all_time_steps, all_rewards, all_labels


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default=None)
    parser.add_argument('--drl_root', type=str, default='../DRL/data')
    parser.add_argument('--drl_abs_root', type=str, default='../r2l/logs/ppo')
    parser.add_argument('--regus_root', type=str, default='../Karel_Script/store/mcts_test/karel_log')
    parser.add_argument('--store_path', type=str, default='store')
    parser.add_argument('--end_timestep', type=float, default=None)
    args = parser.parse_args()

    # load DRL paths
    DRL_root_path = args.drl_root
    DRL_paths = []
    for path in os.listdir(DRL_root_path):
        if args.env_name in path:
            log_path = os.path.join(DRL_root_path, path, 'PPO_1')
            for file in os.listdir(log_path):
                if 'events.out.tfevents' in file:
                    log_path = os.path.join(log_path, file)
                    break
            DRL_paths.append(log_path)

    # load DRL-abs paths
    DRL_abs_root_path = args.drl_abs_root
    DRL_abs_paths = []
    for path in os.listdir(DRL_abs_root_path):
        if args.env_name in path:
            next_name = os.listdir(os.path.join(DRL_abs_root_path, path))[-1]
            log_path = os.path.join(DRL_abs_root_path, path, next_name)
            for file in os.listdir(log_path):
                if 'events.out.tfevents' in file:
                    log_path = os.path.join(log_path, file)
                    break
            DRL_abs_paths.append(log_path)

    # load ReGus paths
    Regus_path = None
    ReGus_root_path = args.regus_root
    for path in os.listdir(ReGus_root_path):
        if args.env_name in path:
            log_path = os.path.join(ReGus_root_path, path)
            for file in os.listdir(log_path):
                if file[-3:] == 'npy':
                    log_path = os.path.join(log_path, file)
                    break
            Regus_path = log_path
            break

    all_time_steps, all_rewards, all_labels = load_data_raw(DRL_paths, DRL_abs_paths, Regus_path)

    if not os.path.exists(os.path.join(args.store_path, args.env_name)):
        os.makedirs(os.path.join(args.store_path, args.env_name))
    var_plot(args, all_time_steps, all_rewards, all_labels, fig_path=os.path.join(args.store_path, args.env_name, 'reward.png'))