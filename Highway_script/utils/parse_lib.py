import argparse

def get_parse():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--task', type=str ,default='highway-fast-v0', \
                        choices=['highway-fast-v0'], \
                        help='task name for karel')
    parser.add_argument('--search_iter', type=str, default='80', \
                        help='search iteration for each sketch')
    parser.add_argument('--store_path', type=str, default='store/mcts_test/highway', \
                        help='root path to store results')
    
    # search
    parser.add_argument('--search_seed_list', type=str, default='0,1000,2000', \
                        help='list of search seeds')
    parser.add_argument('--support_seed_list', type=str, default='40000,70000,10000,50000,60000', \
                        help='list of support seeds')

    # evaluation
    parser.add_argument('--eval_num', type=int, default=5,
                        help='number of evaluation numbers')

    args = parser.parse_args()

    return args