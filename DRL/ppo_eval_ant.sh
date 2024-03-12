python main_eval_karel_ppo_sb3.py --env_name 'stairClimber'


eval_ppo_doorkey_env(){
    python main_eval_karel_ppo_sb3.py \
    --env_name 'doorkey' \
    --model_path 'data/karel_ppo_doorkey/model' \
    --store_path 'karel_doorkey_result_ppo' \
    --eval_mode 'rew'
}