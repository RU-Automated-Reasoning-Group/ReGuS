search_cleanHouse(){
    python mcts_search.py \
    --task 'cleanHouse' \
    --search_iter 500 \
    --store_path 'store/mcts_test/karel_log/cleanHouse' \
    --cost_w 0.04
}

search_fourCorners(){
    python mcts_search.py \
    --task 'fourCorners' \
    --search_iter 500 \
    --store_path 'store/mcts_test/karel_log/fourCorners' \
    --cost_w 0.04
}

search_stairClimber(){
    python mcts_search.py \
    --task 'stairClimber' \
    --search_iter 500 \
    --store_path 'store/mcts_test/karel_log/stairClimber' \
    --cost_w 0.04
}

search_topoff(){
    python mcts_search.py \
    --task 'topOff' \
    --search_iter 500 \
    --store_path 'store/mcts_test/karel_log/topOff' \
    --cost_w 0.04
}

search_randomMaze(){
    python mcts_search.py \
    --task 'randomMaze' \
    --search_iter 500 \
    --store_path 'store/mcts_test/karel_log/randomMaze' \
    --cost_w 0.04
}

search_harvester(){
    python mcts_search.py \
    --task 'harvester' \
    --search_iter 500 \
    --support_seed_list '' \
    --store_path 'store/mcts_test/karel_log/harvester' \
    --cost_w 0.04
}

search_seeder(){
    python mcts_search.py \
    --task 'seeder' \
    --search_iter '600,3500' \
    --store_path 'store/mcts_test/karel_log/seeder_2' \
    --cost_w 0.2
}

search_doorkey(){
    python mcts_search.py \
    --task 'doorkey' \
    --search_iter '600,3500' \
    --store_path 'store/mcts_test/karel_log/doorkey' \
    --cost_w 0.2
}

# search_cleanHouse
# search_fourCorners
# search_stairClimber
# search_topoff
# search_randomMaze
# search_harvester
# search_seeder
search_doorkey