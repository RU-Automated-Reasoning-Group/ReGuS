do_train_pick(){
    python3 r2l.py ppo \
    --env 'FetchHookAbs-v0' \
    --arch 'gru' \
    --save_actor 'Hook_seed_0.pt' \
    --layers 64 \
    --batch_size 6 \
    --num_steps 10000 \
    --prenormalize_steps 100 \
    --timesteps 1000000 \
    --workers 7 \
    --traj_len 100 \
    --logdir './hook_logs' \
    --seed 0
}

do_eval_pick(){
    python r2l.py ppo_eval \
    --env 'FetchPlaceABS-v0' \
    --arch 'gru' \
    --layers 64 \
    --batch_size 6 \
    --num_steps 10000 \
    --prenormalize_steps 100 \
    --timesteps 1000000000 \
    --policy 'pick_and_place_grasp_reward_seed_1.pt' \
    --eval_num 300 \
    --workers 1 \
    --traj_len 500
}

# do_train
# do_train_doorkey
# do_train_2
# do_train_highway
# do_eval_seeder
# do_eval_doorkey
# do_eval_highway
do_train_pick
# do_eval_pick
