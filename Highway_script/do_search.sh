search_highway(){
    python mcts_search_highway.py \
    --task 'highway-fast-v0' \
    --search_iter '80' \
    --search_seed_list '0,1000,2000' \
    --support_seed_list '40000,70000,10000,50000,60000' \
    --store_path 'store/mcts_test/highway_2'
}

search_highway