import argparse

def get_parse():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--tasks', type=str ,default='AntU,AntMaze', \
                        help='task list for ant environment')
    parser.add_argument('--search_iter', type=str, default='80', \
                        help='search iteration for each sketch')
    parser.add_argument('--store_path', type=str, default='store/mcts_test/highway', \
                        help='root path to store results')
    
    # search
    parser.add_argument('--search_seed_list', type=str, default='0,1000,2000', \
                        help='list of search seeds')
    parser.add_argument('--support_seed_list', type=str, default='0,1000,2000,3000,4000', \
                        help='list of support seeds')

    # evaluation
    parser.add_argument('--eval_num', type=int, default=5,
                        help='number of evaluation numbers')

    args = parser.parse_args()

    return args