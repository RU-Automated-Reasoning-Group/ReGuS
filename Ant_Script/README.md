# ReGuS for Ant Environment

In this directory, we provide artifact to test ReGus in [Mujoco Ant Environment](https://gymnasium.farama.org/environments/mujoco/ant/), including Ant U-shaped Maze, Ant S-shaped Maze, Ant $\pi$-shaped Maze and Ant w-shaped Maze. In general, we synthesize a program on U-shaped Maze first and then enhance the prgoram by adding more conditional statements on S-shaped Maze.

## Step-by-step Instruction

We will first introduce how to evaluate the artifact and then summarize the code structure.

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