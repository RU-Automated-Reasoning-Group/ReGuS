# Program Synthesis for Robot Learning


Deep reinforcement learning (RL) has led to encouraging successes in numerous challenging robotics applications. However, the lack of inductive biases to support logic deduction and generalization in the representation of a deep RL model causes it less effective in exploring complex long-horizon robot-control tasks with sparse reward signals. Existing program synthesis algorithms for RL problems inherit the same limitation, as they either adapt conventional RL algorithms to guide program search or synthesize robot-control programs to imitate an RL model. We propose ReGuS, a reward-guided synthesis paradigm, to unlock the potential of program synthesis to overcome the exploration challenges. We develop a novel hierarchical synthesis algorithm with decomposed search space for loops, on-demand synthesis of conditional statements, and curriculum synthesis for procedure calls, to effectively compress the exploration space for long-horizon, multi-stage, and procedural robot-control tasks that are difficult to address by conventional RL techniques. Experiment results demonstrate that ReGuS significantly outperforms state-of-the-art RL algorithms and standard program synthesis baselines on challenging robot tasks including autonomous driving, locomotion control, and object manipulation.

## Getting Started Guide

We recommend machines have at least 16GB memory and 16GB of hard disk space available when building and running Docker images. All benchmarks were tested on Mac Mini 2023 containing Apple M2 Pro CPU and 16GB RAM.

### Requirements

This artifact is built as a Docker image. Before proceeding, ensure Docker is installed. (```sudo docker run hello-world``` will test your installation.) If Docker is not installed, install it via the [official installation guide](https://docs.docker.com/get-docker/). This guide was tested using Docker version 20.10.23, but any contemporary Docker version is expected to work.

### Use Pre-built Docker Image

You might fetch our pre-built Docker image from Docker Hug:
```
docker pull gfc669/pldi_new:latest
```

To launch a shell n docker image:
```
docker run --platform linux/x86_64 -it gfc669/pldi_new
```

Enter root path of code:
```
cd ~/code/ReGuS
```


### Basis Test

To verify the tool operating successfully, entire directory ```Get_Start```.
```
cd Get_Start
```

Enter anaconda environment ```ReGus```:
```
conda activate regus
```

And then run the simple test script
```
python simple_test.py
```

The scipt will test several programs in related environment. A success test will output several warnings that could be ignored on the terminal and more importantly three "test success" word as follow:

```
Simple Test Ant Case: Success
Simple Test Karel Case: Success
Simple Test Highway Case: Success
```

A failure test will output at least one line of ```Test Fail```.  A success test indicates that the system is ready to evaluate artifact.

After the test, please quit the anaconda virtual environment by:
```
conda deactivate
```


## Conclusion of environment

In this section, we provide the general introduction to evaluate our artifacts. Detail instructions are included in each related directories. For proposed ReGuS, we provide the evaluation in 5 popular RL tasks, including Karel, Highway, Ant, Fetch and MiniGrid environments. For each environment, we provide a separate directory to cover all related evaluation cases which will be introduced in following paragraphs. Besides, we also provides artifacts to reproduce baselines results in our papers, including DRL and DRL-abs. Similarly, evaluation of baselines are gathered in respective directories.

### Karel Environment

Directory ```Karel_Script``` contains code of ReGuS method solving Karel tasks (e.g. **Fig.17. of paper**), including cleanHouse, fourCorners, stairClimber, topOff, randomMaze, harvester, seeder and doorkey puzzles. For each puzzle, user could expect visualization results with reward-timestep figures (e.g. refer to **Fig.18. of paper**) showing performance of ReGuS and log file showing searched and success programs synthesized by ReGuS.

### Highway Environment

Directory ```Highway_Script``` encapsulate artifact of ReGuS solving continuous state space task, specifically Highway Driving Environment (e.g. **Left most figure of Fig.21. of paper**). Proposed script will generalize ReGuS performance with reward-timestep figure (e.g. **Fig. 22(a) of paper**) and log files showing synthesized programs by our method.

### Ant Environment

Directory ```Ant_Script``` covers ReGuS method handling Ant task with continuous state space environments, including U-shaped Maze, S-shaped maze, $\pi$-shaped maze and w-shaped maze (e.g. images of mazes are shown in right four of **Fig.21. in paper**). To evaluate ReGuS, we synthesize programs from Ant-U Maze and enhance success program from Ant-U Maze on Ant-S environment. All synthesized complete program will be evaluate on all the four maze puzzles and the result will be present on solved number-time step figure (e.g. **Fig. 22(d) of paper**). Log files showing synthesis footprints will also be stored as result.

### Fetch Environment
Directory ```Fetch_Script` contains the Fetch-Pick&Place and Fetch-Hook environment. These two environments are used in Figure 22 (b)(c). The synthesized programs will be saved and the reward curves can be ploted using the provided script.

### MiniGrid Environment
Directory ```Minigrid_Script``` contains the curriculum synthesis part of ReGuS. Programs synthesized for simple environments can be used as building block for complicated environment. The synthesized program and the environment steps used will be saved as results.

### Baselines

Directory ```DRL``` contains evaluation code for baseline method DRL on Karel, Highway and Ant environments. We mainly applied PPO method with Stable-Baseline 3 framework as DRL and evaluate with same setting as ReGuS. 

Directory ```r2l``` contains evaluation artifacts for baseline method DRL-abs on Karel, Highway and Ant environments. The baseline is modified based on [R2L framework](https://github.com/siekmanj/r2l).

Besides, we also provides a visualization tool to combine result data of baselines and ReGuS and create figures of Karel, Highway and Ant environment (e.g. **Fig.18, 22 in the paper**) in directory ```Visualization```.

## Step-by-step Instruction for Karel

In this directory, we provide artifact to test ReGus in 8 tasks of [Karel Environment](https://compedu.stanford.edu/karel-reader/docs/python/en/chapter1.html), including cleanHouse, fourCorners, stairClimber, topOff, randomMaze, harvester, seeder and doorkey tasks. We will first introduce how to evaluate the artifact and then summarize the code structure.

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

## Step-by-step Instruction for Highway

In this directory, we provide artifact to test ReGus in [Highway Environment](https://github.com/Farama-Foundation/HighwayEnv), especially for highway driving task. In general, we synthesize a program with MCTS Search until an agent execution timestep. 

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

## Step-by-step Instruction for Ant

In this directory, we provide artifact to test ReGus in [Mujoco Ant Environment](https://gymnasium.farama.org/environments/mujoco/ant/), including Ant U-shaped Maze, Ant S-shaped Maze, Ant $\pi$-shaped Maze and Ant w-shaped Maze. In general, we synthesize a program on U-shaped Maze first and then enhance the prgoram by adding more conditional statements on S-shaped Maze. We will first introduce how to evaluate the artifact and then summarize the code structure.

### Synthesis & Evaluation

Before evaluation, please first active conda virtual environment by:
```
conda activate regus
```

If you are under root path of ReGuS, enter ```Ant_Script``` first by:
```
cd Ant_Script
```

And then run the example evaluation code from ```do_search.sh``` as:
```
python ant_search.py --num_exps 1
```

Argument ```--num_exps``` indicates the number of synthesis experiment ReGuS will run. For each synthesis experiment, by default, we search for a success program on *AntU* environment first and continue the synthesis to get an enhanced program on *AntFb* environment. For both first success program and enhanced program, we evaluate them over all 4 environment for 100 iterations respetively. (Noted that in the paper, we evaluate for 1000 times for each environment. We reduce the evaluation iteration here for efficiency). A program with success rate larger than 95% on a specific environment is regarded as success program for such environment. Setting ```num_exps``` to be $n$ will repeat the synthesis with ReGuS with $n$ times.

After the experiment finishes, logs of synthesis on *AntU* and *AntFb* will be created under directory ```store/AntUAntFb/```. Additionally, a solved_env - timesteps figures will be created, where x-axis refers to timesteps used and y-axis refers to number of experiments has been solved by a program. 

**The figure results should be closed to Fig.22(d) of the paper.** (The figures in paper shows the average result on 5 repeated experiments).

### Expected Performance

As expected and could be check from the figure and logs, the success program synthesized from *AntU* would achieve above 95% when testing on *AntU* environment but fail on other 3 environments. The enhanced program from *AntFb* will achieve above 95% on all the 4 environments. 

The expected running time for the entire search of each experiment is 4 hours.

### Code Structure

- ```ant_program_env.py```: Ant environment based on gym
- ```ant_search.py```: High level framework of program synthesis with ReGuS on ant environment.
- ```search_karel_new_while.py```: Program synthesis detail with ReGuS.
- ```dsl.py```: Domain Specific Language of ReGuS program.

## Step-by-step Instruction for Minigrid

To run the code for MiniGrid environments, use conda env regus 2 by running
```
conda activate regus2
```

Install MiniGrid Package
```
cd Minigrid
python3 -m pip install -e .
```

### 1.  Synthesizing a new program for a specific minigrid environment with library

ReGuS is capable of complicated programs based simple programs that synthesized for other environments, i.e. curriculum synthesis. For each environment, represented as an index number, we provide a checkpoint library that contains all synthesized programs for previous environments. Such a library will be used as building blocks for later environments.

```
# for index 0, use
python3 sequence.py --run_one


# for index > 0, use the following command
# substitute [IDX] with the actually number

python3 sequence.py --idx [IDX] --library checkpoints/library_ckpt_[IDX] --library_dict checkpoints/library_dict_ckpt_[IDX] --run_one
```

Different index represents a different MiniGrid environment, shown in the following table

| Index      | Environment                            | Estimated Time    |
| -----------| ---------------------------------------|-------------------|
| 0          | MiniGrid-RandomCrossingS11N5-v0        | 30 mins           |
| 1          | MiniGrid-RandomLavaCrossingS11N5-v0    | 2 mins            |
| 2          | MiniGrid-MultiRoomNoDoor-N6-v0         | 2 mins            |
| 3          | MiniGrid-MultiRoom-N6-v0               | 2 mins            |
| 4          | MiniGrid-LockedRoom-v0                 | 40 mins           |
| 5          | MiniGrid-DoorKey-8x8-v0                | 30 mins           |
| 6          | MiniGrid-PutNearTwoRoom-v0             | 20 mins           |
| 7          | MiniGrid-UnlockPickup-v0               | 15 mins           |

Once the synthesis terminates for the selected in, a text file that has the same name as the environment will be generated in the `results` folder. This text file contains the synthesized program. The output of this program is a detailed log describing the synthesis procedure. However, the output can be very long so output should be discard to `/dev/null` or splited into multiple files such as 

```
python3 sequence.py --idx 1 --library checkpoints/library_ckpt_1 --library_dict checkpoints/library_dict_ckpt_1 --run_one | split -d -b 50m - log.
```

### 2. Full Curriculum Synthesis
Use the following command 
```
python3 sequence.py | split -d -b 50m - log.
```

This is equivalent to running each of the seperated environments in one run. The program will take about 2.5 hours to finish and will produce a synthesized program for each environment. 

The steps used to synthesize the programs can be found in the logs files at the end of searching for each environment. Searching "Moving to next env" to check the number of steps used. An figure called `minigrid.png` will be generated to show the number of environments steps needed to solve the environments.


### Code Structure
- `checkpoints` contains the provided library checkpoint for each environment
- `Minigrid` contains the MiniGrid environment
- `mcts` folder and `mcts_search.py`contain the implementation of the Monte Carlo Tree Search (MCTS) used by ReGus
- `results` folder contains the generated program
- `minigrid_base_dsl.py` specify the action and predicate sets used for each of the environments. This file also implements the Domain Specific Language (DSL) used by ReGuS. This file should be considered as the interface of ReGuS that users will most probabily change for new environents or test new actions. The actually implementation of actions and predicates are defined in the two following locations.
- `Minigrid/minigrid/minigrid_env.py`: This file is contains the base class for all minigrid environments and also contains the implementation of all predicates. For example, the `front_is_clear` predicate is defined in the class method also called `front_is_clear`
- `minigrid_implement` folder contains the actual implementation of the low level actions (what is underhood of actions from a ReGuS program) that directly interact with the environment
- `search.py` and `search2.py` contains the partial program executor for ReGuS DSL and implement the main logic of ReGuS, such as generate new actions within a program or synthesize if/ifelse when necessary. `search2.py` always use multiple seeds to evaluate the performance of a program
- `sequence.py` contains the curriculum synthesis procedure that synthesize program for a new environments using existing programs from previous environments. In this file, we also set hyperparameters for ReGuS for different environment. We also uses some heuristics that the synthesized programs must satisfy to make them more likely to generalize in later environments. ReGuS will possibly generate multiple programs that works for the current environments, however these programs are not equally generalizable for more complex environments.
 
## Step-by-step Instruction for Fetch

To run the code for fetch environments, use conda env regus 2 by running
```
conda activate regus2
```

Set environment variable in bash
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
```

### 1.  Synthesizing a new program for Pick&Place environment 
There are two required arguments. Seed is used to specify different environment initial configuration. Random_seed is is used to control the randomness in the program


seed and random_seed can be changed accordingly.

The estimated running time is 15 minutes. The synthesized program can be found in file "pick.txt" after the synthesis terminates. The output of this program is a is a detailed log showing how the result program is synthesized.
```
python3 search.py --seed 0 --random_seed 0
```

One example program is 
```
WHILE(not (block_at_goal)) { 
        IF(block_below_gripper) { 
            IF(gripper_are_open) { 
                IF(block_inside_gripper) { 
                    close_gripper
                } ELSE { 
                    move_down
                }
            } ELSE { 
                IF(block_inside_gripper) { 
                    move_to_goal
                } ELSE { 
                    open_gripper
                }
            }
        } ELSE { 
            move_to_block
        } 
    idle
} ;; END
```


### 2. Synthesizing a new program for Hook environment

The estimated running time is 2.5 minutes. The synthesized program can be found in file "hook.txt" after the synthesis terminates.
```
python3 search_hook.py --seed 0 --random_seed 0
```

One example program is
```
WHILE(not (block_at_goal)) { 
        IF(hook_grasped) { 
            IF(hook_aligned) { 
                sweep
            } ELSE { 
                align
            }
        } ELSE { 
            pick_up_hook
        } 
    idle
} ;; END
```

### 3. Generate Plot (Figure 22 b&c)
```
python3 generate_plot.py
```

The output are two PDF files which are similar curves for ReGus as in Fig 22 (b) & (c). The trailing horizontal line is missing because ReGuS terminates once it finds the best program.

## 4. Code Structure
- `reskill` folder contains the environment files
- `programskill` folder contains the actual implementation of the low level actions and predicates (what is underhood of actions or predicates from the ReGuS program) that directly interact with the environment
- `robot_dsl.py` & `robot_hook_dsl.py` specify the action and predicate sets used for each of the environments. These two files also implement the Domain Specific Language (DSL) used by ReGuS. These two files should be considered as the interface of ReGuS that users will most probabily change for new environents or test new actions.
- `search.py` contains the partial program executor for ReGuS DSL and implement the main logic of ReGuS, such as generate new actions within a program or synthesize if/ifelse when necessary. 

## 5. Configuring Predicates Used for Synthesis

1. ReGuS predicates interface

    The predicates used for syntehsis for environment can be configured by adjusting the `COND_DICT` dictionary defined at line 39 - 80 of `robot.dsl` in the `Fetch_Script` folder. For example, the `block_at_goal` predicate and its negation can be defined as in the following code block. Users can comment these lines to disable this predicate. New predicates can be defined similarly.

    ```
    COND_DICT = {
        "block_at_goal": k_cond(
            negation=False, cond=k_cond_without_not("block_at_goal")
        ),
        ...
        "not(block_at_goal)": k_cond(
            negation=True, cond=k_cond_without_not("block_at_goal")
        ),
    }
    ```

2. Environment predicate implementation

    However, the definition above is only the high level interface used by the ReGuS synthesis algorithm. The set of all states that satisfy `block_at_goal` is actually provided by the `Fetch-Pic&Place` environment. The `block_at_gaol` function definition is located at line 554 - 557 at file `reskill/rl/envs/fetch_pick_and_place.py` in this folder. User can define new predicate by providing such a function that determines the set of states that the predicate is true or false.

3. ReGuS Abstract State

    The abstract state tracked by ReGuS is defined at line 86 of `robot_dsl.py`. It should contain all predicates used by ReGuS but without negated ones. The constant number (orginally 6) on line 108 should be the number of predicates used by ReGuS and should be updated correctly.

    ```
    class ABS_STATE:
        def __init__(self):
            self.state = {
                ...
                "block_at_goal": None,
                ...
            }
    
    ... 

    def get_abs_state(robot):
        abs_state = ABS_STATE()
        for cond in COND_LIST[:6]:
    ```

## 6. Configuring Actions Used for Synthesis

1. ReGuS action interface

    The actions used for syntehsis for environment can be configured by adjusting the `ACTION_DICT` dictionary defined at line 19 - 26 of `robot.dsl` in the `Fetch_Script` folder. For example, the `move_to_block` action is defined as in the following code block. Users can comment these lines to disable this action. New actions can be defined similarly.

    ```
    ACTION_DICT = {
        ...
        "move_to_block": k_action("move_to_block"),
        ...
    }
    ```

2. Action implementation

    Given a ReGuS action such as `move_to_block`, what effect it will have on the enironment is defined in the `programskill/dsl.py` file. For example, line 96 - 110 is the implementation of `move_to_block`. This ReGuS action will find the position of the block and generate a low level action to move the gripper towards this location. New actions can be defined similarly by accessing the `k.env.obs` as the state from the environement.

## Step-by-step Instruction for Custom Environment

To run ReGuS on custom environment, user needs to provide environment definition (e.g ```karel``` folder in ```Karel_Script```) and domain specific language definition (e.g. ```dsl_karel_new.py``` scipt in ```Karel_Script```). In most cases, user could directly use existed dsl script for new environment, only to import custom environment inside dsl script instead of existed environment.