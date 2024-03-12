from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import pdb

path = 'data/highway_ppo_seed0/PPO_1/events.out.tfevents.1695498299.sss.cs.rutgers.edu.2822265.0'
results = [e for e in summary_iterator(path)]
pdb.set_trace()
print('o?')