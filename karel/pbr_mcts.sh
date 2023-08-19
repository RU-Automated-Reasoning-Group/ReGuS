karel_mcts_seeder(){
    python run_mcts_search.py \
        --task_name 'seeder' \
        --sub_goals '1' \
        --search_iter 1000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_mcts_doorkey(){
    python run_mcts_search.py \
        --task_name 'doorkey' \
        --sub_goals '0.5,1' \
        --search_iter 1000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_mcts_stairClimber(){
    python run_mcts_search.py \
        --task_name 'stairClimber' \
        --sub_goals '1' \
        --search_iter 1000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_mcts_cleanHouse(){
    python run_mcts_search.py \
        --task_name 'cleanHouse' \
        --sub_goals '1' \
        --search_iter 1000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_mcts_topOff(){
    python run_mcts_search.py \
        --task_name 'topOff' \
        --sub_goals '1' \
        --search_iter 1000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_mcts_randomMaze(){
    python run_mcts_search.py \
        --task_name 'randomMaze' \
        --sub_goals '1' \
        --search_iter 1000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_mcts_fourCorners(){
    python run_mcts_search.py \
        --task_name 'fourCorners' \
        --sub_goals '1' \
        --search_iter 1000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_mcts_harvester(){
    python run_mcts_search.py \
        --task_name 'harvester' \
        --sub_goals '1' \
        --search_iter 1000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

# karel_mcts_seeder
# karel_mcts_stairClimber
# karel_mcts_cleanHouse
# karel_mcts_topOff
# karel_mcts_randomMaze
# karel_mcts_fourCorners
karel_mcts_harvester