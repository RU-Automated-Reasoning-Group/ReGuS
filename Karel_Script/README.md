# ReGuS for Karel Environment

In this directory, we provide an artifact to test ReGuS across 8 tasks of the [Karel Environment](https://compedu.stanford.edu/karel-reader/docs/python/en/chapter1.html), including cleanHouse, fourCorners, stairClimber, topOff, randomMaze, harvester, seeder, and doorkey tasks.

## Step-by-step Instruction

We will first introduce how to evaluate the artifact and then summarize the code structure.

### Synthesis & Evaluation

Before evaluation, please activate the conda virtual environment by:
```
conda activate regus
```

If you are under the root path of ReGuS, enter the ```Karel_Script``` directory first by:
```
cd Karel_Script
```

The file ```do_search.sh``` provides example evaluation code for specific environments as:
```
python mcts_search.py --task [env_name] --num_exps 1
```

We recommend running each environment in a separate process rather than directly running the shell file ```do_search.sh```, as sequentially running all the Karel environments will be highly time-consuming.

The argument ```--num_exps``` indicates the number of synthesis experiments ReGuS will run. For each synthesis experiment, by default, we apply ReGuS to search for programs and execute them in the related Karel environment. After the experiments finish, logs of synthesis will be created under the directory ```store/mcts/karel_log/[env_name]```. Moreover, for each Karel environment, a reward-timesteps figure will be generated, where the x-axis refers to timesteps used and the y-axis refers to the reward achieved for each synthesized program.

**The figure results should be close to Fig.18 of the paper.** (The figures in the paper show the average result of 5 repeated experiments).

### Expected Performance

The ```store_demo``` directory contains the logs and figure results we obtained by running the artifact. As expected, across all Karel environments, ReGuS should find a program that achieves a reward of 1. The expected time cost for each environment is shown as below:

| Index      | Environment    | Estimated Time    |
| -----------| ---------------|-------------------|
| 0          | CleanHouse     | 30 mins           |
| 1          | FourCorners    | 1 mins            |
| 2          | StairClimber   | 3 mins            |
| 3          | TopOff         | 1 mins            |
| 4          | RandomMaze     | 6 mins            |
| 5          | Harvester      | 45 mins           |
| 6          | Seeder         | 40 mins           |
| 7          | DoorKey        | 100 mins          |

### Code Structure

- ```karel```: Karel environment.
- ```mcts```: MCTS Search to synthesize high-level sketches.
- ```dsl_karel_new.py```: Domain Specific Language for ReGuS program on Karel Environment.
- ```mcts_search.py```: High-level framework for ReGuS algorithm.
- ```search_karel_new.py```: Low-level details of ReGuS algorithm.