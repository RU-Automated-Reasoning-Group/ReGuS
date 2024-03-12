from highway_general.robot import HighwayRobot

def do_highway_test():
    config_set = {'observation': {'type': 'TimeToCollision'},
                'lanes_count': 8,
                'vehicles_density': 0.2}
    robot = HighwayRobot('highway-fast-v0', seed=0, view='TTC', config_set=config_set)