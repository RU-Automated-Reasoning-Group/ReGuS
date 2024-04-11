# ReGuS for Highway Environment

In this directory, we provide an artifact to test ReGuS in the [Highway Environment](https://github.com/Farama-Foundation/HighwayEnv), specifically for the highway driving task. Generally, we synthesize a program using MCTS Search until an agent execution timestep.

## Step-by-step Instruction

We will first introduce how to evaluate the artifact and then summarize the code structure.

### Synthesis & Evaluation

Before evaluation, please activate the conda virtual environment by:
```
conda activate regus
```

If you are under the root path of ReGuS, enter the ```Highway_script``` directory first by:
```
cd Highway_script
```

Then run the example evaluation code from ```do_search.sh``` as:
```
python mcts_search_highway.py --num_exps 1
```

The argument ```--num_exps``` indicates the number of synthesis experiments ReGuS will run. For each synthesis experiment, by default, we apply ReGuS to search for programs and execute them in the highway driving environment. After the experiment finishes, logs of synthesis on the *Highway* will be created under the directory ```store/mcts_test/highway```. Additionally, a reward-timesteps figure will be generated, where the x-axis refers to timesteps used and the y-axis refers to the reward achieved for each synthesized program.

**The figure results should be closed to Fig.22(a) of the paper.** (The figures in the paper show the average result of 5 repeated experiments).

### Expected Performance

The ```store_demo``` directory contains the logs and figure results we obtained by running the artifact. As expected and can be verified from the last line of logs, the best program synthesized should achieve a reward of above 14. The best average reward achieved over 3 different experiments should be above 15.

The expected running time for the entire search of each experiment is 4 hours.

### Code Structure

- ```highway_general```: Highway environment based on Gym
- ```mcts```: MCTS Search to synthesize high-level sketches
- ```dsl_highway_all.py```: Domain Specific Language for ReGuS program on the Highway Environment.
- ```mcts_search_highway.py```: High-level framework for the ReGuS algorithm.
- ```search_highway_new.py```: Low-level details of the ReGuS algorithm.