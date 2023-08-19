import argparse

def get_karel_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, \
                        choices=['stairClimber', 'cleanHouse', 'topOff', 'randomMaze', 'fourCorners', \
                                 'harvester', 'seeder', 'doorkey'], \
                        default='randomMaze', \
                        help='task name for karel')
    parser.add_argument('--search_seed', type=int, default=0, help='random seed set for search')
    parser.add_argument('--more_seed', type=str, default=None, help='list of more seeds with string separate by ","')
    parser.add_argument('--sub_goals', type=str, default='1', help='list of reward goals for search')
    parser.add_argument('--search_iter', type=int, default=2000, help='number of max search iteration')
    parser.add_argument('--max_stru_cost', type=int, default=20, help='limit of structure cost')
    parser.add_argument('--stru_weight', type=float, default=0.2, help='weight for structure cost for synthesis score')

    return parser.parse_args()