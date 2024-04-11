# Deep Reinforcement Learning Baseline

In this directory, we provide an artifact to test the DRL baseline in Ant, Highway, and Karel environments. In our paper and this artifact, we leverage PPO from [Stable Baseline 3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) as the DRL method.

## Step-by-step Instruction

We will first introduce how to evaluate the artifact and then summarize the code structure.

### Train DRL

Before training, please activate the conda virtual environment by:

```
conda activate regus
```

If you are under the root path of ReGuS, enter the ```DRL``` directory first by:

```
cd DRL
```

For a specific environment, users can test DRL with the following command:
```
python main_ppo_sb3.py --env_name [environment name] --num_exps 1
```

The argument ```--num_exps``` indicates the number of training experiments DRL will run. By default, the training log and DRL model will be stored under the directory ```data/[environment_ppo_seed]```.

We provide example commands in ```ppo_train.sh``` for Karel, Highway, and Ant environments.

For Karel environments, example commands include:
```
# Karel
python main_ppo_sb3.py --env_name 'doorkey' --num_exps 1
python main_ppo_sb3.py --env_name 'seeder' --num_exps 1
python main_ppo_sb3.py --env_name 'harvester' --num_exps 1
python main_ppo_sb3.py --env_name 'cleanHouse' --num_exps 1
python main_ppo_sb3.py --env_name 'randomMaze' --num_exps 1
python main_ppo_sb3.py --env_name 'stairClimber' --num_exps 1
python main_ppo_sb3.py --env_name 'topOff' --num_exps 1
python main_ppo_sb3.py --env_name 'fourCorners' --num_exps 1
```

For the Highway environment, example commands include:
```
# highway
python main_ppo_sb3.py --env_name 'highway' --num_exps 1
```

For Ant environments, example commands include:
```
# ant
python main_ppo_sb3.py --env_name 'AntU' --num_exps 1
python main_ppo_sb3.py --env_name 'AntFb' --num_exps 1
python main_ppo_sb3.py --env_name 'AntFg' --num_exps 1
python main_ppo_sb3.py --env_name 'AntMaze' --num_exps 1
```

To check the results of DRL training, users can access the log file with TensorBoard as follows
```
cd data/[environment_ppo_seed]

# For example: logdir=PPO_1
python -m tensorboard.main --logdir=PPO_[execute_id]
```
The default log results will be shown at ```http://localhost:6006```. The logs and models run by us can be accessed from ```data_demo```. **The figure results should be closed to Fig.18 and Fig.22 of the paper.** (The figures in the paper show the average result of 5 repeated experiments).

### Evaluate DRL

After training, users can load the stored model to evaluate its performance in environments. Example commands are provided in ```ppo_eval.sh```.sh as:
```
python main_eval_ppo_sb3.py --env_name [environment name] --model_path [path]
```

## Code Structure
- ```highway_general_self```: Highway environment based on Gymnasium.
- ```karel```: Karel environment based on Gymnasium.
- ```ant```: Ant environment based on Gymnasium and MuJoCo.
- ```main_ppo_sb3.py```: Script to train PPO in related environments.
- ```main_eval_ppo_sb3.py```: Script to evaluate trained PPO in related environments.