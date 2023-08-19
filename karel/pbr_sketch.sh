karel_seeder(){
    python run_pbr_sketch.py \
        --task_name 'seeder' \
        --sub_goals '1' \
        --search_iter 8000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_doorkey(){
    python run_pbr_sketch.py \
        --task_name 'doorkey' \
        --sub_goals '0.5,1' \
        --search_iter 10000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_stairClimber(){
    python run_pbr_sketch.py \
        --task_name 'stairClimber' \
        --sub_goals '1' \
        --search_iter 8000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_cleanHouse(){
    python run_pbr_sketch.py \
        --task_name 'cleanHouse' \
        --sub_goals '1' \
        --search_iter 8000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_topOff(){
    python run_pbr_sketch.py \
        --task_name 'topOff' \
        --sub_goals '1' \
        --search_iter 8000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_randomMaze(){
    python run_pbr_sketch.py \
        --task_name 'randomMaze' \
        --sub_goals '1' \
        --search_iter 8000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_fourCorners(){
    python run_pbr_sketch.py \
        --task_name 'fourCorners' \
        --sub_goals '1' \
        --search_iter 8000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

karel_harvester(){
    python run_pbr_sketch.py \
        --task_name 'harvester' \
        --sub_goals '1' \
        --search_iter 8000 \
        --max_stru_cost 20 \
        --stru_weight 0.2
}

# karel_seeder
# karel_doorkey
# karel_stairClimber
# karel_cleanHouse
# karel_topOff
# karel_randomMaze
# karel_fourCorners
karel_harvester