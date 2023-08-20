# NOTE: each node is associated with a sketch
import subprocess

from dsl import *
from gym_minigrid.robot import MiniGridRobot


count = 0
seed_list = []
for i in range(0, 1000):
    robot = MiniGridRobot('MiniGrid-MultiRoom-N6-v0', i)
    env = robot.env
    left_pos = env.agent_pos + (-1 * env.right_vec)
    left_cell = env.grid.get(*left_pos)
    if left_cell is not None and left_cell.type == 'wall':
        count += 1
        seed_list.append(i)

print(seed_list)
for i in range(0, len(seed_list)):
    seed = seed_list[i]
    eval_seed1 = seed_list[i + 1]
    eval_seed2 = seed_list[i + 2]
    eval_seed3 = seed_list[i + 3]
    eval_seed4 = seed_list[i + 4]
    subprocess.run(f"python3 current/tree/search.py {str(seed)} {str(eval_seed1)} {str(eval_seed2)} {str(eval_seed3)} {str(eval_seed4)} > seeds/{str(seed)}.txt", shell=True)