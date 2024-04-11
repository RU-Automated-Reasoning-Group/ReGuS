# Deep Reinforment Learning with Abstract State (DRL-Abs) Baseline

In this directory, we provide an artifact to test the DRL-abs baseline in Ant, Highway, and Karel environments. In our paper and this artifact, we leverage [recurrent reinforcement learning (r2l)](https://github.com/siekmanj/r2l) as the baseline to train an agent with an abstract state of the related environments.

## Step-by-step Instruction

We will first introduce how to evaluate the artifact and then summarize the code structure.

### Train DRL-abs

Before training, activate the conda virtual environment as follows:
```
conda activate regus
```

If you are under the root path of ReGuS (```ReGus```), enter the r2l directory first by:
```
cd r2l
```

For a specific environment, users can test r2l with the following command:
```
python r2l.py train --env [environment name] --num_exps 1
```

The argument ```--num_exps``` indicates the number of training experiments DRL will run. By default, the training log will be stored under the directory ```logs/ppo/[environment-seed]```, and the model will be stored under the directory ```store```.

We provide example commands in ```train_r2l.sh``` for Karel, Highway, and Ant environments.

For Karel environments, example commands include:
```
# Karel
python r2l.py train --env 'seeder' --num_exps 1
python r2l.py train --env 'doorkey' --num_exps 1
python r2l.py train --env 'harvester' --num_exps 1
python r2l.py train --env 'cleanHouse' --num_exps 1
python r2l.py train --env 'randomMaze' --num_exps 1
python r2l.py train --env 'stairClimber' --num_exps 1
python r2l.py train --env 'topOff' --num_exps 1
python r2l.py train --env 'fourCorners' --num_exps 1
```

For the Highway environment, example commands include:
```
# highway
python r2l.py train --env 'highway' --num_exps 1
```

For Ant environments, example commands include:
```
# ant
python r2l.py train --env 'AntU' --num_exps 1
python r2l.py train --env 'AntFb' --num_exps 1
python r2l.py train --env 'AntFg' --num_exps 1
python r2l.py train --env 'AntMaze' --num_exps 1
```

To check the results of DRL training, users can access the log file with TensorBoard as follows:
```
cd logs/ppo

# For example: logdir=highway-v0-0
python -m tensorboard.main --logdir=[environment-seed]
```
The default log results will be shown at ```http://localhost:6006```. The logs and models run by us can be accessed from ```logs_demo``` and ```store_demo```. **The figure results should be close to Fig.18 and Fig.22 of the paper.** (The figures in the paper show the average result of 5 repeated experiments).



### Evaluate DRL-abs

After training, users can load the stored model to evaluate its performance in environments. Example commands are provided in ```eval_r2l.sh``` as:
```
python r2l.py eval --env [environment name] --policy [policy path]
```

## Code Structure
- ```highway_general```: Highway environment based on Gymnasium.
- ```karel```: Karel environment based on Gymnasium.
- ```ant```: Ant environment based on Gymnasium and MuJoCo.
- ```policies```: Model architecture of actor-critic framework.
- ```spinup```: Detail strategy of the reinforcement learning method.
- ```r2l.py```: Script with the main function to train and evaluate the r2l method.