# ReGuS for Highway Environment

In this directory, we provide artifact to test ReGus in [Highway Environment](https://github.com/Farama-Foundation/HighwayEnv), especially for highway driving task. In general, we synthesize a program with MCTS Search until an agent execution timestep.

## Step-by-step Instruction

We will first introduce how to evaluate the artifact and then summarize the code structure.

### Synthesis & Evaluation

Before evaluation, please first active conda virtual environment by:
```
conda activate regus
```

If you are under root path of ReGuS, enter ```Highway_script``` first by:
```
cd Highway_script
```

And then run the example evaluation code from ```do_search.sh``` as:
```
python mcts_search_highway.py --num_exps 1
```

Argument ```--num_exps``` indicates the number of synthesis experiment ReGuS will run. For each synthesis experiment, by default, we apply ReGuS to search for programs and execute them on highway drive environment. After the experiment finishes, logs of synthesis on *Highway* will be created under directory ```store/mcts_test/highway```. Additionally, a reward-timesteps figure will be generated, where x-axis refers to timesteps used and y-axis refers to reward achieved for each synthesized program. 

**The figure results should be closed to Fig.22(a) of the paper.** (The figures in paper shows the average result on 5 repeated experiments).

### Expected Performance

Directory ```store_demo``` contains the logs and figure result we get by running the artifact. As expected and could be checked from the last line of logs, the best program synthesized should achieve reward to be above 14. And the best average reward achieved over 3 different experiment should be above 15.

The expected running time for the entire search of each experiment is 4 hours.

### Code Structure

- ```highway_general```: Highway environment based on gym
- ```mcts```: MCTS Search to synthesize high level sketch
- ```dsl_highway_all.py```: Domain Specific Language for ReGuS program on Highway Environment.
- ```mcts_search_highway.py```: Hig level framework for ReGuS algorithm.
- ```search_highway_new.py```: Low level details of ReGuS algorithm.