# ReGuS for Ant Environment

In this directory, we provide an artifact to test ReGuS in the [Mujoco Ant Environment](https://gymnasium.farama.org/environments/mujoco/ant/), including Ant U-shaped Maze, Ant S-shaped Maze, Ant $\pi$-shaped Maze, and Ant W-shaped Maze. Generally, we synthesize a program on the U-shaped Maze first and then enhance the program by adding more conditional statements on the S-shaped Maze.

## Step-by-step Instruction

We will first introduce how to evaluate the artifact and then summarize the code structure.

### Synthesis & Evaluation

Before evaluation, please activate the conda virtual environment by:
```
conda activate regus
```

If you are under the root path of ReGuS (```[YourPath]/ReGuS```), enter the ```Ant_Script``` directory first by:
```
cd Ant_Script
```

Then run the example evaluation code from ```do_search.sh``` as:
```
python ant_search.py --num_exps 1
```

The argument```--num_exps``` indicates the number of synthesis experiments ReGuS will run. For each synthesis experiment, by default, we search for a successful program on the *AntU* environment first and continue the synthesis to get an enhanced program on the *AntFb* environment. For both the first successful program and the enhanced program, we evaluate them over all 4 environments for 100 iterations respectively. (Note that in the paper, we evaluated for 1000 times for each environment. We have reduced the evaluation iteration here for efficiency). A program with a success rate greater than 95% on a specific environment is regarded as a successful program for that environment. Setting ```num_exps``` to $n$ will repeat the synthesis with ReGuS $n$ times.

After the experiment finishes, logs of synthesis on *AntU* and *AntFb* will be created under the directory ```store/AntUAntFb/```. Additionally, a solved_env-timesteps figure will be created, where the x-axis refers to timesteps used and the y-axis refers to the number of experiments that have been solved by a program.

**The figure results should be closed to Fig.22(d) of the paper.**  (The figures in the paper show the average result of 5 repeated experiments).

### Expected Performance

As expected and as can be verified from the figures and logs, the successful program synthesized from *AntU* will achieve above 95% when tested on the *AntU* environment but fail on the other three environments. The enhanced program from *AntFb* will achieve above 95% on all 4 environments.

The expected running time for the entire search of each experiment is 4 hours.

### Code Structure

- ```ant_program_env.py```: Ant environment based on Gymnasium
- ```ant_search.py```: High level framework of program synthesis with ReGuS on Ant environment.
- ```search_karel_new_while.py```: Program synthesis detail with ReGuS.
- ```dsl.py```: Domain Specific Language of ReGuS program.