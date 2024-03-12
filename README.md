# Program Synthesis for Robot Learning


Deep reinforcement learning (RL) has led to encouraging successes in numerous challenging robotics applications. However, the lack of inductive biases to support logic deduction and generalization in the representation of a deep RL model causes it less effective in exploring complex long-horizon robot-control tasks with sparse reward signals. Existing program synthesis algorithms for RL problems inherit the same limitation, as they either adapt conventional RL algorithms to guide program search or synthesize robot-control programs to imitate an RL model. We propose ReGuS, a reward-guided synthesis paradigm, to unlock the potential of program synthesis to overcome the exploration challenges. We develop a novel hierarchical synthesis algorithm with decomposed search space for loops, on-demand synthesis of conditional statements, and curriculum synthesis for procedure calls, to effectively compress the exploration space for long-horizon, multi-stage, and procedural robot-control tasks that are difficult to address by conventional RL techniques. Experiment results demonstrate that ReGuS significantly outperforms state-of-the-art RL algorithms and standard program synthesis baselines on challenging robot tasks including autonomous driving, locomotion control, and object manipulation.

## Getting Started Guide

We recommend machines have at least 16GB memory and 16GB of hard disk space available when building and running Docker images. All benchmarks were tested on Mac Mini 2023 containing Apple M2 Pro CPU and 16GB RAM.

### Requirements

This artifact is built as a Docker image. Before proceeding, ensure Docker is installed. (```sudo docker run hello-world``` will test your installation.) If Docker is not installed, install it via the [official installation guide](https://docs.docker.com/get-docker/). This guide was tested using Docker version 20.10.23, but any contemporary Docker version is expected to work.

### Use Pre-built Docker Image

You might fetch our pre-built Docker image from Docker Hug:
```
docker pull gfc669/pldi_ubuntu_latest:latest
```

To launch a shell n docker image:
```
docker run --platform linux/x86_64 -it gfc669/pldi_ubuntu_latest
```

### Basis Test

To verify the tool operating successfully, run [TODO]

## Step-by-step Instructions

In this section, we provide the general introduction to evaluate our artifacts. Detail instructions are included in each related directories. For proposed ReGuS, we provide the evaluation in 5 popular RL tasks, including Karel, Highway, Ant, Fetch and MiniGrid environments. For each environment, we provide a separate directory to cover all related evaluation cases which will be introduced in following paragraphs. Besides, we also provides artifacts to reproduce baselines results in our papers, including DRL and DRL-abs. Similarly, evaluation of baselines are gathered in respective directories.

### Karel Environment

Directory ```Karel_Script``` contains code of ReGuS method solving Karel tasks (e.g. **Fig.17. of paper**), including cleanHouse, fourCorners, stairClimber, topOff, randomMaze, harvester, seeder and doorkey puzzles. For each puzzle, user could expect visualization results with reward-timestep figures (e.g. refer to **Fig.18. of paper**) showing performance of ReGuS and log file showing searched and success programs synthesized by ReGuS.

### Highway Environment

Directory ```Highway_Script``` encapsulate artifact of ReGuS solving continuous state space task, specifically Highway Driving Environment (e.g. **Left most figure of Fig.21. of paper**). Proposed script will generalize ReGuS performance with reward-timestep figure (e.g. **Fig. 22(a) of paper**) and log files showing synthesized programs by our method.

### Ant Environment

Directory ```Ant_Script``` covers ReGuS method handling Ant task with continuous state space environments, including U-shaped Maze, S-shaped maze, $\pi$-shaped maze and w-shaped maze (e.g. images of mazes are shown in right four of **Fig.21. in paper**). To evaluate ReGuS, we synthesize programs from Ant-U Maze and enhance success program from Ant-U Maze on Ant-S environment. All synthesized complete program will be evaluate on all the four maze puzzles and the result will be present on solved number-time step figure (e.g. **Fig. 22(d) of paper**). Log files showing synthesis footprints will also be stored as result.

### Fetch Environment

### MiniGrid Environment
