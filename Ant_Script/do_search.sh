search_highway(){
    python ant_search.py \
    --task 'AntU,AntFb' \
    --search_iter '80' \
    --search_seed_list '2000' \
    --support_seed_list '0,1000,2000,3000,4000' \
    --store_path 'store/highway'
}

search_highway