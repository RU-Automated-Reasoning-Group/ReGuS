# Program Synthesis for Robot Learning

Deep reinforcement learning (RL) has led to encouraging successes in numerous challenging robotics applications. However, the lack of inductive biases to support logic deduction and generalization in the representation of deep RL models makes them less effective in exploring complex long-horizon robot-control tasks with sparse reward signals. Existing program synthesis algorithms for RL problems inherit this limitation, as they either adapt conventional RL algorithms to guide program search or synthesize robot-control programs to imitate an RL model. We propose ReGuS, a reward-guided synthesis paradigm, to unlock the potential of program synthesis in overcoming exploration challenges. We develop a novel hierarchical synthesis algorithm with a decomposed search space for loops, on-demand synthesis of conditional statements, and curriculum synthesis for procedure calls, effectively compressing the exploration space for long-horizon, multi-stage, and procedural robot-control tasks that are difficult to address with conventional RL techniques. Experimental results demonstrate that ReGuS significantly outperforms state-of-the-art RL algorithms and standard program synthesis baselines on challenging robot tasks including autonomous driving, locomotion control, and object manipulation.

## Getting Started Guide

We recommend machines have at least 16GB of memory and 16GB of hard disk space available when building and running Docker images. All benchmarks were tested on a Mac Mini 2023 containing an Apple M2 Pro CPU and 16GB of RAM.

### Requirements

This artifact is built as a Docker image. Before proceeding, ensure Docker is installed. ((```sudo docker run hello-world``` will test your installation.) If Docker is not installed, please install it via the [official installation guide](https://docs.docker.com/get-docker/). This guide was tested using Docker version 20.10.23, but any contemporary Docker version is expected to work.

### Use Pre-built Docker Image

You can fetch our pre-built Docker image from Docker Hub:

```
docker pull gfc669/pldi_new:latest
```

To launch a shell in the docker image:
```
docker run --platform linux/x86_64 -it gfc669/pldi_new
```

Enter the root path of the code:
```
cd ~/code/ReGuS
```


### Basis Test

To verify the tool is operating successfully, navigate to the entire directory ```Get_Start```.
```
cd Get_Start
```

Enter the anaconda environment ```ReGus```:
```
conda activate regus
```

Then run the simple test script：
```
python simple_test.py
```

The script will test several programs in related environments. A successful test will output several warnings that can be ignored on the terminal and, more importantly, three "test success" messages as follows:
```
Simple Test Ant Case: Success
Simple Test Karel Case: Success
Simple Test Highway Case: Success
```

A failed test will output at least one line of ```Test Fail```. A successful test indicates that the system is ready to evaluate the artifact.

After the test, please exit the anaconda virtual environment by:
```
conda deactivate
```


## Step-by-step Instructions

In this section, we provide a general introduction to evaluate our artifacts. Detailed instructions are included in each related directory. For the proposed ReGuS, we provide the evaluation in 5 popular RL tasks, including Karel, Highway, Ant, Fetch, and MiniGrid environments. For each environment, we provide a separate directory to cover all related evaluation cases which will be introduced in the following paragraphs. In addition, we also provide artifacts to reproduce baseline results in our paper, including DRL and DRL-abs. Similarly, evaluations of baselines are gathered in respective directories.

### Conda Virtual Environment

Generally, we have two conda environments in the docker, ```regus``` and ```regus2```. To run ReGuS for Karel, Highway and Ant environments, please activate ```regus```; to run ReGus for Fetch and MiniGrid environments, please activate ```regus2```.

### Karel Environment

The directory ```Karel_Script``` contains the code of the ReGuS method solving Karel tasks (e.g., **Fig.17. of paper**), including cleanHouse, fourCorners, stairClimber, topOff, randomMaze, harvester, seeder, and doorkey puzzles. For each puzzle, users can expect visualization results with reward-timestep figures (e.g., refer to **Fig.18. of paper**) showing the performance of ReGuS and a log file showing searched and successful programs synthesized by ReGuS.

### Highway Environment

The directory ```Highway_Script`` encapsulates the artifact of ReGuS solving the continuous state space task, specifically the Highway Driving Environment (e.g., **Leftmost figure of Fig.21. of the paper**). The proposed script will generalize ReGuS performance with a reward-timestep figure (e.g., **Fig. 22(a) of the paper**) and log files showing synthesized programs by our method.

### Ant Environment

The directory ```Ant_Script``` covers the ReGuS method handling Ant tasks with continuous state space environments, including U-shaped Maze, S-shaped maze, $\pi$-shaped maze, and W-shaped maze (e.g., images of mazes are shown in the right four of **Fig.21. in the paper**). To evaluate ReGuS, we synthesize programs from the Ant-U Maze and enhance the success program from the Ant-U Maze on the Ant-S environment. All synthesized complete programs will be evaluated on all four maze puzzles, and the result will be presented in a solved number-time step figure (e.g., **Fig. 22(d) of the paper**). Log files showing synthesis footprints will also be stored as a result.

### Fetch Environment

The directory ```Fetch_Script``` contains the Fetch-Pick&Place and Fetch-Hook environments. These two environments are used in Figure 22 (b)(c). The synthesized programs will be saved, and the reward curves can be plotted using the provided script.

### MiniGrid Environment

The directory ```Minigrid_Script``` contains the curriculum synthesis part of ReGuS. Programs synthesized for simple environments can be used as building blocks for complicated environments. The synthesized program and the environment steps used will be saved as results.

### Baselines

The directory ```DRL``` contains evaluation code for the baseline method DRL on Karel, Highway, and Ant environments. We mainly applied the PPO method with the Stable-Baselines 3 framework as DRL and evaluated with the same setting as ReGuS.

The directory ```r2l``` contains evaluation artifacts for the baseline method DRL-abs on Karel, Highway, and Ant environments. The baseline is modified based on the [R2L framework](https://github.com/siekmanj/r2l).

Besides, we also provide a visualization tool to combine the result data of baselines and ReGuS and create figures of the Karel, Highway, and Ant environment (e.g.,  **Fig.18, 22 in the paper**) in the directory ```Visualization```.

## Step-by-step Instruction for Custom Environment

In this section, we introduce two examples to demonstrate how to apply ReGuS to a new environment developed by user. In the first example, we assume the Highway environment to be the new environment and take the Karel environment as an existing template. We will modify modules from the Karel script step-by-step to adapt ReGuS to the Highway environment. In the second example, we introduce a new Karel task, Karel UpDown-StairClimber, and show how to reuse code from the Karel script to run ReGuS in this task.

### Highway Environment

Using the Karel script as an existing template, we aim to adapt the code for the Highway environment. We start by copying the Karel script and renaming the new script as ```ReGuS Script```:
```
ReGus Script
|    dsl_karel.py
|    search_karel.py
|    mcts_search.py
|    do_search.sh
|
|____karel
|    |    [files for Karel environment]
|
|____mcts
|    |    MCTS_search_tree.py
|    |    search_alg.py
|    |
|____utils
|    |    [files for utils]
```

#### Environment Definition

In the above ```ReGuS Script```, the ```karel``` directory defines the environment and tasks. Since the Highway environment has a completely different definition, this module needs rewriting:

```
ReGus Script
|    dsl_karel.py
|    search_karel.py
|    mcts_search.py
|    do_search.sh
|
|____highway (updated)
|    |    [files from Highway GitHub]
|    |    dsl.py
|    |    robot.py
|
|____mcts
|    |    MCTS_search_tree.py
|    |    search_alg.py
|    |
|____utils
|    |    [files for utils]
```

We download the environment directory from [Highway GitHub](https://github.com/Farama-Foundation/HighwayEnv) and rename it as ```highway```, which includes code for transition details. In order to run ReGuS, we need to define perceptions and actions accessible by the generated program. We define perceptions to track vehicles in the current lane, left lane, and right lane as *front_is_clear*, *left_is_clear* and *right_is_clear*, respectively. Based on "time to crash" observations in the highway environment, we implement perception labels as class *h_cond_without_not* in ```highway/dsl.py```. Additionally, we create a class *h_cond* to handle the **not** logical operation on perception.

For actions, we implement *faster* and *slower* to increase and decrease the velocity of the ego vehicle, respectively, and *idle* to maintain the current velocity. We also implement *Lane_left* and *Lane_right* to control the ego vehicle to navigate into the left and right lanes, respectively. These actions are included in class *h_action* in ```highway/dsl.py```.


As a conclusion, the code file ```highway/dsl.py``` contains:
```
# highway dsl

# perceptions: front_is_clear, left_is_clear, right_is_clear
class h_cond_without_not:
    ...

# handle not logic
class h_cond:
    ...

# actions: faster, slower, idle, lane_left, lane_right
class h_action:
    ...
```

Then in  ```highway/robot.py```, we rewrite environment initialization (reset), transition (execute_single_action, execute_single_cond), and reward definition (custom_reward) on top of the highway implementation to make program perceptions and actions compatible. A summary of this file is shown below:
```
# highway robot

class HighwayRobot:
    def __init__(self, task, seed):
        self.env = gym.make(task, render_mode='rgb_array')
        ...

    # initialization
    def reset(self):
        ...
    
    # transition given action
    def execute_single_action(self, action):
        ...

    # results for a perception
    def execute_single_cond(self, cond):
        ...

    # reward function on top of highway
    def custom_reward(self):
        ...
```

For detailed implementation of the highway environment, please refer to  ```Highway_script/highway_general/dsl.py``` and ```Highway_script/highway_general/robot.py```.

#### Domain Specific Language (DSL)

As noted in the paper, we need to define DSL for program structure. Most parts of DSL are general and not directly related to the environment, thus we reuse ```dsl_karel.py``` with minor modifications and rename it as ```dsl_highway.py```. The updated file structure is displayed below:
```
ReGus Script
|    dsl_highway.py (updated)
|    search_karel.py
|    mcts_search.py
|    do_search.sh
|
|____highway (updated)
|    |    [files from Highway GitHub]
|    |    dsl.py
|    |    robot.py
|
|____mcts
|    |    MCTS_search_tree.py
|    |    search_alg.py
|    |
|____utils
|    |    [files for utils]
```

Candidate action statements and perception conditions for program generation are specific to the environment and need to be rewritten. Specifically, we revise the ```ACTION_DICT``` in ```dsl_highway.py``` to include new action statements as follows:
```
from highway_general.dsl import h_action
ACTION_DICT = {
    'lane_left'   : h_action(0),
    'idle'        : h_action(1),
    'lane_right'  : h_action(2),
    'faster'      : h_action(3),
    'slower'      : h_action(4)
}
```

Moreover, we rewrite ```COND_DICT``` and ```ABS_STATE``` in ```dsl_highway.py``` for new perception labels as:
```
from highway_general.dsl import h_cond, h_cond_without_not
# perception conditions
COND_DICT = {
    'front_is_clear_3'    : h_cond(negation=False, cond=h_cond_without_not('front_is_clear_3')),
    'left_is_clear_3'     : h_cond(negation=False, cond=h_cond_without_not('left_is_clear_3')),
    'right_is_clear_3'    : h_cond(negation=False, cond=h_cond_without_not('right_is_clear_3')),
    'all_true'          : h_cond(negation=False, cond=h_cond_without_not('all_true')),
    'not(front_is_clear_3)'    : h_cond(negation=True, cond=h_cond_without_not('front_is_clear_3')),
    'not(left_is_clear_3)'     : h_cond(negation=True, cond=h_cond_without_not('left_is_clear_3')),
    'not(right_is_clear_3)'    : h_cond(negation=True, cond=h_cond_without_not('right_is_clear_3')),
}

# abstract state
class ABS_STATE:
    def __init__(self):
        self.state = {
            'front_is_clear_3'    : None,
            'left_is_clear_3'     : None,
            'right_is_clear_3'    : None,
        }
```

**DSL for Program Generation** Other code of ```dsl_highway.py``` can be reused by replacing all ```k_cond``` from ```dsl_karel.py``` into ```h_cond```. 

#### ReGuS algorithm

The other files, including ```search_karel.py```, ```mcts_search.py``` and directory ```MCTS```, implement the core components of the ReGuS framework. Only minor modifications are required to adapt these codes for use in new environments. We will detail these necessary adjustments in the rest of this section.

```
ReGuS Script
|    dsl_highway.py (updated)
|    search_highway.py (updated)
|    mcts_search.py (updated)
|    do_search.sh
|
|____highway (updated)
|    |    [files from Highway GitHub]
|    |    dsl.py
|    |    robot.py
|
|____mcts (updated)
|    |    MCTS_search_tree.py
|    |    search_alg.py
|    |
|____utils
|    |    [files for utils]
```

First, we rename  ```search_karel.py``` into ```search_highway.py```. To adapt this file for the Highway environment, users will need to modify lines 11 and 12 to import the environment definitions specific to the Highway setup. Additionally, lines 81-86 should be adjusted to initialize the Highway environment appropriately. The rest of the code in ```search_high.py``` can be reused without further changes. (For more details, refer to ```Highway_script/search_highway_new.py```).
```
# search_highway.py

# line 11-12
from dsl_highway_all import *
from highway_general.robot import HighwayRobot

# line 81
self.robot_store = {self.seed: HighwayRobot(self.task, seed=self.seed, view=view_mode, config_set=config_set)}

# line 84
self.robot_store[e] = HighwayRobot(self.task, seed=e, view=view_mode, config_set=config_set)

# line 86
self.eval_robot_store[e] = HighwayRobot(self.task, seed=e, view=view_mode, config_set=config_set)
```

For ```mcts_search.py```, user needs to modify line 4 to import DSL of Highway environment as
```
# mcts_search.py

# line 4
from dsl_highway imoprt *
```

Similarly, for ```mcts/MCTS_search_tree.py```, user needs to modify line 3 to import DSL of Highway environment. And for ```mcts/search_alg.py```, user needs to modify line 12 to import program synthesis code as below (More details in ```Highway_script/mcts```)
```
# line 3 of mcts/MCTS_search_tree.py
from dsl_highway imoprt *

# line 12 of mcts/search_alg.py
from search_highway_new import Node
```

With the following command:
```
python mcts_search.py --num_exps 1
```
ReGuS would be able to run on Highway environment.

### Karel UpDown-StairClimber

Similar to the Highway Environment adaptation, we again use the Karel script as the existing template and reuse the code for the Karel UpDown-StairClimber task.
```
Karel Script
|    dsl_karel.py
|    search_karel.py
|    mcts_search.py
|    do_search.sh
|
|____karel
|    |    checker.py
|    |    dsl.py
|    |    generator.py
|    |    karel.py
|    |    robot.py
|    |    [other files for karel environment]
|
|____mcts
|    |    MCTS_search_tree.py
|    |    search_alg.py
|    |
|____utils
|    |    [files for utils]
```

#### Environment Definition

While in Karel StairClimber task the agent is only required to go upstair, in Karel UpDown-StairClimber the agent is required to go upstair first and then go downstair. However, Karel UpDown-StairClimber and Karel StairClimber share the same low-level karel environment definition and reward function. Users only need to add task generation in ```karel/generator.py``` and modify ```karel/robot.py``` to import the new task generation. The modifications can be implemented as follows:
```
# karel/generator.py
    def generate_single_state_up_down_stair_climber(self, h=12, w=12, wall_prob=0.1, env_task_metadata={}):
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[0, :, 4] = True
        s[h-1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w-1, 4] = True

        random.seed(self.seed)

        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0,   0, '-',   0,   0,   0,   0, '-'],
            ['-',   0,   0,   0,   0, '-', '-', '-',   0,   0,   0, '-'],
            ['-',   0,   0,   0, '-', '-',   0, '-', '-',   0,   0, '-'],
            ['-',   0,   0, '-', '-',   0,   0,   0, '-', '-',   0, '-'],
            ['-',   0, '-', '-',   0,   0,   0,   0,   0, '-', '-', '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ]
        ...

# karel/robot.py
class KarelRobot:
    def state_init(self, gen, task):
        ...
        elif task == 'upDown':
            gen_function = gen_function_base + 'up_down_stair_climber'
        ...
```

Besides the environment definition, users can reuse all other parts of the ReGuS code. The resulting file structure is as follows:
```
Karel Script
|    dsl_karel.py
|    search_karel.py
|    mcts_search.py
|    do_search.sh
|
|____karel
|    |    checker.py (updated)
|    |    dsl.py
|    |    generator.py (updated)
|    |    karel.py
|    |    robot.py (updated)
|    |    [other files for karel environment]
|
|____mcts
|    |    MCTS_search_tree.py
|    |    search_alg.py
|    |
|____utils
|    |    [files for utils]
```

We provide the example of Karel UpDown-StairClimber in ```Karel_Script/karel```. Users could uncomment related codes in ```Karel_Script/karel/generator.py``` and ```Karel_Script/karel/robot.py``` and run the experiment as:
```
python mcts_search.py --task 'upDown' --num_exps 1
```