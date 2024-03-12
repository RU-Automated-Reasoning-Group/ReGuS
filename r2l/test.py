from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import pdb

path = 'logs/ppo/karel-seeder-0/55e5ba-seed0/events.out.tfevents.1710186248.sss.cs.rutgers.edu.555376.0'
results = [e for e in summary_iterator(path)]
pdb.set_trace()
print('o?')