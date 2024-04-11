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

In this section, we take the Karel environment as an example to illustrate how to apply ReGuS to a new environment and domain-specific language (Suppose the Karel environment is the custom environment).

### Environment (nonreuseable)

To apply ReGuS for Karel, the first thing a user needs to do is to add the definition of the environment. For example, inside ```Karel_Script```, the directory  ```karel``` contains the abstract environment defining the environment map, actions the robot could take, states the environment will return, and the transaction of the environment based on action.

Because different environments might have totally different action and state spaces, environment definition code could not be reused for a custom environment. For example, the  **Karel** directory is totally different compared to the  **Highway** directory.

### Domain Specific Language (DSL)

Based on an environment, we need to define the DSL for program structure as well as actions and perceptions that could be accessed by the program.

#### DSL for Program Structure (reuseable)

For the Karel environment, a user might want to define statements including:
- *loop* with **WHILE** statement to control repeated actions
- *branch* with **If-Then-Else** statement for the robot to select different actions related to different perceptions. 

In ```dsl_karel_new.py``` of the ```Karel_Script``` directory, we define general statements as *S* class, while loop and branch are defined in *WHILE*, *IF* and *IFELSE* classes.

Additionally, a user would need:
- *Action* to control robot.
- *Condition* for loop and branch to differentiate different states.

In ```dsl_karel_new.py```, classes *C* and *B* are general definitions for action and condition respectively.

Based on the above DSL, a user also needs to implement the program expansion as class *Program*. In most cases, the DSL for program structure could be reused, and related code of classes inside ```dsl_karel_new.py``` could be directly applied to a custom environment.

#### Action & Perception (nonreuseable)

Besides the DSL for program structure, a user also needs to define detailed high-level actions and perception conditions (e.g., *ACTION_DICT*, *COND_DICT*, class *ABS_STATE* in ```dsl_karel_new.py```). These implementations are directly related to the environment and cannot be reused for a custom environment.

### ReGuS Framework (reuseable)

The ReGuS framework mainly contains high-level sketch search and low-level program search. We apply MCTS search for sketch generation as in the ```MCTS``` directory, and the program generation details are included in ```search_karel_new.py```. Both these codes could be safely reused with a change of DSL importing. For example, for the Karel environment, we add:
```
from dsl_karel_new import *
from karel.robot import KarelRobot
```
to ```search_karel_new.py```.