import os

prob = 0.2

for s in range(100, 101):
    # os.system(f"python3 r2l.py ppo --env 'FetchPlaceABS-v0' --arch 'gru' --save_actor 'pick_and_place_grasp_reward_{prob}_seed_{s}_final.pt' --layers 64 --batch_size 6 --num_steps 10000 --prenormalize_steps 100 --timesteps 1000000 --workers 7 --traj_len 100 --logdir './final_logs/grasp_reward_slippery_{prob}' --seed {s}")
    os.system(f"python3 r2l.py ppo --env 'FetchHookAbs-v0' --arch 'gru' --save_actor 'Hook_seed_{s}.pt' --layers 64 --batch_size 6 --num_steps 10000 --prenormalize_steps 100 --timesteps 1000000 --workers 7 --traj_len 100 --logdir './hook_logs' --seed {s}")