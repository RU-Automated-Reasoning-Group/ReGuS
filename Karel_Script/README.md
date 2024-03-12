# ReGuS for Karel Environment

In this directory, we provide artifact to test ReGus in 8 tasks of [Karel Environment](https://compedu.stanford.edu/karel-reader/docs/python/en/chapter1.html), including cleanHouse, fourCorners, stairClimber, topOff, randomMaze, harvester, seeder and doorkey tasks.

## Step-by-step Instruction

We will first introduce how to evaluate the artifact and then summarize the code structure.

### Synthesis & Evaluation

Before evaluation, please first active conda virtual environment by:
```
conda activate regus
```

If you are under root path of ReGuS, enter ```Karel_Script``` first by:
```
cd Karel_Script
```

The file ```do_search.sh``` provides example evaluation code for specific environment as:
```
python mcts_search.py --task [env_name] --num_exps 1
```
We recommend to run each environment on separate process rather than directly run shell file ```do_search.sh```, for the reason that sequentially running all the karel environments will result in high time consuming.

Argument ```--num_exps``` indicates the number of synthesis experiment ReGuS will run. For each synthesis experiment, by default, we apply ReGuS to search for programs and execute them on related Karel environment. After the experiments finish, logs of synthesis will be created under directory ```store/mcts/karel_log/[env_name]```. Moreover, for each karel environment, a reward-timesteps figure will be generated, where x-axis refers to timesteps used and y-axis refers to reward achieved for each synthesized program. 

**The figure results should be closed to Fig.18 of the paper.** (The figures in paper shows the average result on 5 repeated experiments).

### Expected Performance

Directory ```store_demo``` contains the logs and figure result we get by running the artifact. As expect, the all the karel environment, ReGuS could find a program achieve reward 1. The expected time cost for each environment is shown as below:

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

- ```karel```: Karel environment
- ```mcts```: MCTS Search to synthesize high level sketch
- ```dsl_karel_new.py```: Domain Specific Language for ReGuS program on Karel Environment.
- ```mcts_search.py```: High level framework for ReGuS algorithm.
- ```search_karel_new.py```: Low level details of ReGuS algorithm.